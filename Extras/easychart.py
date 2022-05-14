from fastapi import FastAPI
from easycharts import ChartServer
from easyschedule import EasyScheduler

server = FastAPI()

@server.on_event('startup')
async def setup():
    server.charts = await ChartServer.create(
        server,
        charts_db="test"
    )

    await server.charts.create_dataset(
        "test",
        labels=['a', 'b', 'c', 'd'],
        dataset=[1,2,3,4]
    )