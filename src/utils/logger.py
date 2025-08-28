import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.layout import Layout
from rich.live import Live

# Custom theme for sign language detection project
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "dim blue",
        "data": "magenta",
        "model": "blue",
        "training": "green",
        "test": "yellow",
        "realtime": "bright_cyan",
        "detection": "bright_green",
        "bbox": "orange3",
        "loss": "red3",
        "accuracy": "green3",
    }
)


class SignLanguageLogger:
    """Enhanced logger with Rich formatting for Sign Language Detection project."""

    def __init__(self, name: str = "sign_language", level: str = "INFO"):
        self.console = Console(theme=custom_theme)
        self.name = name
        self.level = getattr(logging, level.upper())

        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup the logging configuration with rich formatting."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(self.level)

        # File handler for persistent logs
        log_file = (
            self.logs_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(rich_handler)
        self.logger.addHandler(file_handler)

    def info(self, message: str, **kwargs):
        """Log info message with rich formatting."""
        self.logger.info(f"[info]{message}[/info]", **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with rich formatting."""
        self.logger.warning(f"[warning]{message}[/warning]", **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with rich formatting."""
        self.logger.error(f"[error]{message}[/error]", **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message with rich formatting."""
        self.logger.info(f"[success]✅ {message}[/success]", **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with rich formatting."""
        self.logger.debug(f"[debug]{message}[/debug]", **kwargs)

    def data(self, message: str, **kwargs):
        """Log data-related message with rich formatting."""
        self.logger.info(f"[data]📊 {message}[/data]", **kwargs)

    def model(self, message: str, **kwargs):
        """Log model-related message with rich formatting."""
        self.logger.info(f"[model]🤖 {message}[/model]", **kwargs)

    def training(self, message: str, **kwargs):
        """Log training-related message with rich formatting."""
        self.logger.info(f"[training]🏋️ {message}[/training]", **kwargs)

    def test(self, message: str, **kwargs):
        """Log test-related message with rich formatting."""
        self.logger.info(f"[test]🧪 {message}[/test]", **kwargs)

    def realtime(self, message: str, **kwargs):
        """Log realtime-related message with rich formatting."""
        self.logger.info(f"[realtime]📹 {message}[/realtime]", **kwargs)

    def detection(self, message: str, **kwargs):
        """Log detection-related message with rich formatting."""
        self.logger.info(f"[detection]🎯 {message}[/detection]", **kwargs)

    def print_panel(self, title: str, content: str, style: str = "blue"):
        """Print content in a rich panel."""
        panel = Panel(content, title=title, style=style, border_style=style)
        self.console.print(panel)

    def print_table(self, title: str, headers: list, rows: list):
        """Print data in a rich table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")

        for header in headers:
            table.add_column(header, style="cyan")

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        self.console.print(table)

    def print_status(self, status: str, message: str, style: str = "blue"):
        """Print a status message with icon."""
        status_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "loading": "⏳",
            "done": "🎉",
            "data": "📊",
            "model": "🤖",
            "training": "🏋️",
            "test": "🧪",
            "realtime": "📹",
            "detection": "🎯",
        }

        icon = status_icons.get(status, "•")
        self.console.print(f"[{style}]{icon} {message}[/{style}]")

    def create_progress(self, description: str = "Processing..."):
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

    def create_training_progress(self, loss_string, type='Train'):
        """Create a specialized progress bar for training."""
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{type} Progress"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )

    def print_banner(self):
        """Print the Sign Language Detection project banner."""
        banner = """
             ███╗   ██╗ ██████╗ ██████╗ ██████╗ ██╗   ██╗
            ████╗  ██║██╔═══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
           ██╔██╗ ██║██║   ██║██║  ██║██║  ██║ ╚████╔╝
          ██║╚██╗██║██║   ██║██║  ██║██║  ██║  ╚██╔╝
         ██║ ╚████║╚██████╔╝██████╔╝██████╔╝   ██║
        ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝

╔═══════════════════════════════════════════════════════════╗
║  🤟 Sign Language Detection with DETR                     ║
║  🎯 Real-time Hand Sign Recognition                       ║
║  🏋️  DETR (Detection Transformer) Model                    ║
║  📊 Advanced Computer Vision Pipeline                     ║
╚═══════════════════════════════════════════════════════════╝
        """
        self.console.print(Panel(banner, style="bold cyan", border_style="blue", expand=False))

    def print_model_summary(self, model_info: Dict[str, Any]):
        """Print model architecture summary."""
        table = Table(title="🤖 DETR Model Configuration", show_header=True, header_style="bold blue")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")

        for key, value in model_info.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def print_dataset_info(self, dataset_info: Dict[str, Any]):
        """Print dataset information."""
        table = Table(title="📊 Dataset Information", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        for key, value in dataset_info.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def print_detection_results(self, detections: list):
        """Print detection results in a formatted table."""
        if not detections:
            self.console.print("[yellow]No detections found[/yellow]")
            return

        table = Table(title="🎯 Detection Results", show_header=True, header_style="bold green")
        table.add_column("Class", style="cyan")
        table.add_column("Confidence", style="yellow")
        table.add_column("BBox (x1,y1,x2,y2)", style="orange3")

        for detection in detections:
            table.add_row(
                detection.get("class", "Unknown"),
                f"{detection.get('confidence', 0):.3f}",
                f"({detection.get('bbox', [0,0,0,0])[0]:.1f}, "
                f"{detection.get('bbox', [0,0,0,0])[1]:.1f}, "
                f"{detection.get('bbox', [0,0,0,0])[2]:.1f}, "
                f"{detection.get('bbox', [0,0,0,0])[3]:.1f})"
            )

        self.console.print(table)

    def print_training_metrics(self, epoch: int, train_loss: float, test_loss: float = None, lr: float = None):
        """Print training metrics in a formatted way."""
        metrics_text = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        if test_loss is not None:
            metrics_text += f" | Test Loss: {test_loss:.4f}"
        if lr is not None:
            metrics_text += f" | LR: {lr:.2e}"

        self.console.print(f"[training]{metrics_text}[/training]")

    def capture(self, message: str, **kwargs):
        """Log capture-related message with rich formatting."""
        self.logger.info(f"[realtime]📸 {message}[/realtime]", **kwargs)

    def capture_success(self, class_name: str, image_count: int, **kwargs):
        """Log successful image capture."""
        self.console.print(f"[success]✅ Captured {class_name} image #{image_count}[/success]")

    def capture_error(self, class_name: str, error: str, **kwargs):
        """Log capture error with rich formatting."""
        self.console.print(f"[error]❌ Failed to capture {class_name}: {error}[/error]")

    def capture_class_start(self, class_name: str, total_images: int):
        """Print when starting to capture a new class."""
        self.console.print(f"\n[bold cyan]🎯 Starting capture for class: {class_name}[/bold cyan]")
        self.console.print(f"[dim]Target: {total_images} images[/dim]")

    def capture_session_start(self, classes: list, images_per_class: int, sleep_time: int):
        """Print capture session information."""
        session_info = [
            ["Total Classes", str(len(classes))],
            ["Images per Class", str(images_per_class)],
            ["Total Images", str(len(classes) * images_per_class)],
            ["Sleep Time", f"{sleep_time}s"],
            ["Classes", ", ".join(classes)],
        ]
        self.print_table("📸 Image Capture Session", ["Parameter", "Value"], session_info)

    def capture_session_complete(self, total_captured: int, total_classes: int):
        """Print capture session completion."""
        self.console.print(f"\n[bold green]🎉 Capture session completed![/bold green]")
        self.console.print(f"[success]✅ Total images captured: {total_captured}[/success]")
        self.console.print(f"[success]✅ Classes processed: {total_classes}[/success]")

    def create_capture_progress(self, total_images: int, class_name: str):
        """Create a progress bar specifically for image capture."""
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Capturing {class_name}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )


# Global logger instance
logger = SignLanguageLogger()


def get_logger(name: str = "sign_language") -> SignLanguageLogger:
    """Get a logger instance."""
    return SignLanguageLogger(name)