import sys
import os
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import torch
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text


class DataLoaderHandler:
    """Rich handler for data loading operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.progress = None
        
    def create_data_progress(self, description: str = "Loading Data"):
        """Create progress bar for data loading."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return self.progress
    
    def log_dataset_stats(self, dataset_info: Dict[str, Any]):
        """Display dataset statistics."""
        table = Table(title="📊 Dataset Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in dataset_info.items():
            table.add_row(str(key), str(value))
            
        self.console.print(table)
    
    def log_transform_info(self, transforms_info: List[str]):
        """Display transform information."""
        transforms_text = "\n".join([f"• {transform}" for transform in transforms_info])
        panel = Panel(
            transforms_text,
            title="🔄 Data Transforms",
            style="blue",
            border_style="blue"
        )
        self.console.print(panel)


class TrainingHandler:
    """Rich handler for training operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.epoch_progress = None
        self.batch_progress = None
        self.current_epoch = 0
        self.metrics_history = []
        
    def start_training(self, total_epochs: int, batches_per_epoch: int):
        """Initialize training progress bars."""
        self.epoch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Epoch Progress"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )
        
        self.batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Batch Progress"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        
        # Create layout for both progress bars
        layout = Layout()
        layout.split_column(
            Layout(Panel(self.epoch_progress, border_style="green"), name="epoch"),
            Layout(Panel(self.batch_progress, border_style="blue"), name="batch")
        )
        
        return layout
    
    def update_epoch_metrics(self, epoch: int, train_loss: float, test_loss: float = None, 
                           lr: float = None, additional_metrics: Dict[str, float] = None):
        """Update and display epoch metrics."""
        metrics = {
            "Epoch": epoch,
            "Train Loss": f"{train_loss:.6f}",
        }
        
        if test_loss is not None:
            metrics["Test Loss"] = f"{test_loss:.6f}"
        if lr is not None:
            metrics["Learning Rate"] = f"{lr:.2e}"
        if additional_metrics:
            metrics.update({k: f"{v:.6f}" for k, v in additional_metrics.items()})
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        # Create metrics table
        table = Table(title=f"📈 Training Metrics - Epoch {epoch}", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in metrics.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
    
    def log_loss_components(self, loss_dict: Dict[str, float], epoch: int, batch: int):
        """Log detailed loss components."""
        table = Table(title=f"🎯 Loss Components - Epoch {epoch}, Batch {batch}", 
                     show_header=True, header_style="bold red")
        table.add_column("Loss Component", style="cyan")
        table.add_column("Value", style="red")
        
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            table.add_row(loss_name, f"{loss_value:.6f}")
        
        self.console.print(table)
    
    def save_checkpoint_status(self, checkpoint_path: str, epoch: int):
        """Display checkpoint save status."""
        panel = Panel(
            f"💾 Model checkpoint saved at epoch {epoch}\n📁 Path: {checkpoint_path}",
            title="Checkpoint Saved",
            style="green",
            border_style="green"
        )
        self.console.print(panel)
    
    def create_training_progress(self):
        """Create a specialized progress bar for training."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[red]Train Loss: {task.fields[train_loss]}"),  # Dynamic loss info
            TextColumn("[yellow]Test Loss: {task.fields[test_loss]}"),  # Dynamic loss info
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )


class ModelHandler:
    """Rich handler for model operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def log_model_architecture(self, model_info: Dict[str, Any]):
        """Display model architecture information."""
        table = Table(title="🤖 DETR Model Architecture", show_header=True, header_style="bold blue")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="yellow")
        
        for key, value in model_info.items():
            table.add_row(str(key), str(value))
        
        self.console.print(table)
    
    def log_model_loading(self, model_path: str, success: bool = True):
        """Log model loading status."""
        if success:
            self.console.print(f"[green]✅ Model loaded successfully from {model_path}[/green]")
        else:
            self.console.print(f"[red]❌ Failed to load model from {model_path}[/red]")
    
    def log_parameters_count(self, total_params: int, trainable_params: int):
        """Display parameter count information."""
        table = Table(title="📊 Model Parameters", show_header=True, header_style="bold blue")
        table.add_column("Parameter Type", style="cyan")
        table.add_column("Count", style="yellow")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Non-trainable Parameters", f"{total_params - trainable_params:,}")
        
        self.console.print(table)


class DetectionHandler:
    """Rich handler for detection operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def log_detections(self, detections: List[Dict], frame_id: Optional[int] = None):
        """Display detection results."""
        title = "🎯 Detection Results"
        if frame_id is not None:
            title += f" - Frame {frame_id}"
            
        if not detections:
            self.console.print(f"[yellow]{title}: No detections found[/yellow]")
            return
        
        table = Table(title=title, show_header=True, header_style="bold green")
        table.add_column("Class", style="cyan")
        table.add_column("Confidence", style="yellow")
        table.add_column("BBox (x1,y1,x2,y2)", style="orange3")
        
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            table.add_row(
                detection.get('class', 'Unknown'),
                f"{detection.get('confidence', 0):.3f}",
                f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})"
            )
        
        self.console.print(table)
    
    def log_inference_time(self, inference_time: float, fps: Optional[float] = None):
        """Display inference timing information."""
        timing_info = f"⏱️  Inference Time: {inference_time:.3f}ms"
        if fps is not None:
            timing_info += f" | FPS: {fps:.1f}"
        
        self.console.print(f"[bright_cyan]{timing_info}[/bright_cyan]")


class TestHandler:
    """Rich handler for testing operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def log_test_results(self, test_metrics: Dict[str, float]):
        """Display test results."""
        table = Table(title="🧪 Test Results", show_header=True, header_style="bold yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        self.console.print(table)
    
    def create_test_progress(self, total_samples: int):
        """Create progress bar for testing."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Testing Progress"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )


@contextmanager
def rich_training_context(console: Console = None):
    """Context manager for rich training display."""
    handler = TrainingHandler(console)
    try:
        yield handler
    finally:
        if handler.epoch_progress:
            handler.epoch_progress.stop()
        if handler.batch_progress:
            handler.batch_progress.stop()


@contextmanager
def rich_data_context(console: Console = None):
    """Context manager for rich data loading display."""
    handler = DataLoaderHandler(console)
    try:
        yield handler
    finally:
        if handler.progress:
            handler.progress.stop()


def create_detection_live_display(console: Console = None):
    """Create a live display for real-time detection."""
    console = console or Console()
    
    layout = Layout()
    layout.split_column(
        Layout(Panel("📹 Real-time Detection", border_style="cyan"), name="header", size=3),
        Layout(Panel("Waiting for detections...", border_style="green"), name="detections"),
        Layout(Panel("⏱️ Performance: --", border_style="yellow"), name="performance", size=3)
    )
    
    return Live(layout, console=console, refresh_per_second=10) 