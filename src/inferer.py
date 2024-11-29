import json
from typing import List
import ollama
import socket
import asyncio

from src.chroma import ChromaDatabase

class TcpInferer:
    def __init__(self, host, port, ollamaPort, terminator='<sobadd>'):
        self.host = host
        self.port = port
        self.ollamaPort = ollamaPort
        self.terminator = terminator
        self.ollama = ollama.AsyncClient(host='http://localhost:11434')
        # details of our chromadb
        print(f'TcpInferer Initialized - {host}:{port}')

        self.chroma = ChromaDatabase(host="localhost", port=8000, collectionName="vectordb")


    def display_response_performance(self, part):
        pass


    def compress_prompt(self, messages):
        pass


    async def await_user_prompt(self, reader: asyncio.StreamReader):
        buffer = bytearray()
        # Read until we reach the Terminator - <sobadd>
        while True:
            data = await reader.read(1024)
            stringValue = data.decode('utf-8')
            buffer.extend(data)
            # Read till the Terminator
            if (self.terminator in stringValue):
                msg = buffer.decode()
                msg = msg[:-len(self.terminator)]
                return msg
            
    
    async def update_user_question_for_rag(self, convo):
        if len(convo["tags"]) == 0: return convo["messages"]

        print(convo["tags"])

        question_template = """Given the following conversation and a follow up question, rephrase the Follow Up Question to be a standalone question and do not add any additional text.
        Chat History:
        {}
        Follow Up Question:
        {}
        Standalone Question:"""

        chat_history = "\n".join(["{}: {}".format(m["role"], m["content"]) for m in convo["messages"][:-1]])
        follow_up_question = "{}: {}".format(convo["messages"][-1]["role"], convo["messages"][-1]["content"])

        # Generate summarized prompt - "phi3:mini" or "qwen2:1.5b-instruct-q6_K"
        response = await self.ollama.generate(model="qwen2:1.5b-instruct-q6_K",
                                              prompt=question_template.format(chat_history, follow_up_question), 
                                              stream=False, 
                                              keep_alive="15m")
        
        
        # Parse the summarized response from the llm
        parts = response['response'].split(":\n\n")
        summarized_prompt = parts[1] if len(parts) > 1 else response['response']
        print("LLM Summarized Response: ", summarized_prompt)

        # Find k-neighbours
        neighbours = self.chroma.QueryPrompt(summarized_prompt, 5, convo["tags"])

        # Update last question
        summarized_template = """Given the following supporting information and a question, attempt to answer the question to the best of your abilities.
        Supporting information:
        {}
        Question:
        {}
        Standalone Question:"""

        original_qn = convo["messages"][-1]["content"]
        supporting_info = "\n\n".join(neighbours["documents"][0])
        convo["messages"][-1]["content"] = summarized_template.format(supporting_info, original_qn)

        return convo["messages"]


    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Wait for user prompt
        convoBytes = await self.await_user_prompt(reader)
        print("Received response")

        # Convert byte array into Python Object (kinda json-like)
        convo = json.loads(convoBytes)

        # If conversation tags exist, retrieve supporting information from ChromaDB
        convo_history = await self.update_user_question_for_rag(convo)

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Perform our query and write the response back to the golang server - phi3:mini or qwen2:1.5b-instruct-q6_K
        async for part in await self.ollama.chat(model='qwen2:1.5b-instruct-q6_K', messages=convo_history, stream=True, keep_alive="15m"):
            chunk = part['message']['content']
            writer.write(chunk.encode('utf-8'))
            await writer.drain()

        # Display Performance - Update parameter afterwards
        self.display_response_performance(part)

        # Close the writer once we are done
        await writer.drain()
        print("Done draining")
        writer.close()
        await writer.wait_closed()
        print("Done closing")


    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        async with server:
            await server.serve_forever()
        