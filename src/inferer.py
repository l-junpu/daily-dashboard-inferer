import asyncio
import json
import socket

class TcpInferer:
    def __init__(self, host, port, ollamaPort, terminator='<sobadd>'):
        self.host = host
        self.port = port
        self.ollamaPort = ollamaPort
        self.terminator = terminator
        # details of our chromadb
        print(f'TcpInferer Initialized - {host}:{port}')


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


    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Wait for user prompt
        messages = await self.await_user_prompt(reader)

        # Compress prompt
        compressed_prompt = self.compress_prompt(messages)

        # Convert byte array into Python Object (kinda json-like)
        data = json.loads(messages)

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Perform some ollama stuff

        # Display Performance - Update parameter afterwards
        self.display_response_performance(0)

        # Close the writer once we are done
        writer.close()
        await writer.wait_closed()

        # Debug printing
        print(messages)


    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        async with server:
            await server.serve_forever()
        