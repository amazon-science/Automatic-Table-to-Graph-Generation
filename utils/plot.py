from sqlalchemy import create_engine
from sqlalchemy_schemadisplay import create_schema_graph
from PIL import Image
# Use the sqlalchemy_metadata attribute to get metadata in sqlalchemy format.

def plot_rdb_dataset_schema(dataset, output_path: str):
    output_name = f"{output_path}.pdf"
    output_jpg_name = f"{output_path}.png"
    sa_metadata = dataset.sqlalchemy_metadata
    # import ipdb; ipdb.set_trace()
    graph = create_schema_graph(engine=create_engine('sqlite://'),  # Use a temporary in-memory sqlite db.
                            metadata=sa_metadata,
                            show_datatypes=True,                # The image would show datatypes
                            show_indexes=True,                  # The image would show index names and unique constraints
                            rankdir='LR',                       # From left to right, instead of top to bottom
                            concentrate=False)                  # Don't try to join the relation lines together
    print(f"Writing the schema graph to a file {output_name}...")
    graph.write_pdf(output_name)
    graph.write_png(output_jpg_name)