from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select

def engine_creator(db_url: str):
    try:
        engine = create_engine(db_url, connect_args={'connect_timeout': 5})
        engine.connect().close()
        return engine
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return None

def read_data_from_table(engine, table_name: str):
    if not engine:
        return []
    try:
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        query = select(table)
        with engine.connect() as connection:
            result = connection.execute(query)
            return [dict(row) for row in result.mappings().all()]
    except Exception as e:
        print(f"❌ Table '{table_name}' error: {e}")
        return []
