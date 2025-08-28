from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.console import Console
import time

def create_training_progress(console=None):
    """Create a specialized progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[red]{task.fields[loss_info]}"),  # Dynamic loss info
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console or Console(),
    )

# Example usage
console = Console()
with create_training_progress(console) as progress:
    # Add task with initial description and custom field
    task_id = progress.add_task(
        "[green]Training Epoch 1/10", 
        total=100,
        loss_info="Loss: 0.0000"  # Custom field for dynamic text
    )
    
    for x in range(100): 
        # Update with new loss info
        current_loss = 1.0 / (x + 1)  # Simulated decreasing loss
        progress.update(
            task_id, 
            advance=1,
            loss_info=f"Loss: {current_loss:.4f}"  # Update the custom field
        )
        time.sleep(0.1)  # Simulate work