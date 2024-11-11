import asyncio
from src.inferer import TcpInferer

if __name__ == "__main__":
    inferer = TcpInferer(host='localhost', port=7060, ollamaPort=11434)
    asyncio.run( inferer.run() )