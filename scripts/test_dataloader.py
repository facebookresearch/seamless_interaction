from fairseq2.logging import get_log_writer
from fairseq2.recipes.utils.setup import setup_gangs
from utils.dataset import load_seamless_next_dataset

log = get_log_writer(__name__)

# Create a gang
_, gangs = setup_gangs(
    log=log,
    tp_size=1,
    monitored=False,
)
dp_gang = gangs["dp"]

# Load the dataset
dataset = load_seamless_next_dataset.load("./configs/test_dataloader_yaml.yaml")

# Create a reader
reader = dataset.create_reader(
    gang=dp_gang,
    batch_size=8,
    batch_shuffle_window=100,
    segment_shuffle_window=1000,
    num_prefetch=1,  # Configure prefetching for better throughput
    mismatch_tolerance_seconds=0.1,  # Tolerance for feature duration mismatches
    seed=42,  # Seed for shuffling
)

# Iterate over batches
for batch in reader:
    # Process batch
    print(batch)
    pass
