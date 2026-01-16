"""Workspace Page - Unified view for a dataset with Plot, Export, and Sites tabs.

This page provides a tabbed interface for working with a specific dataset:
1. Plot - Visualize TEMPO data maps
2. Export - Export data to Excel formats
3. Sites - View and manage sites within the dataset bounds
"""

import flet as ft
from pathlib import Path
from typing import Optional
import asyncio
import xarray as xr

from ..theme import Colors, Spacing
from ..components.widgets import SectionCard, StatusLogPanel
from ...storage.database import Database
from ...storage.models import Dataset, Site
from ...core.plotter import MapPlotter
from ...core.exporter import DataExporter


class WorkspacePage(ft.Container):
    """Unified workspace for a dataset with multiple tabs."""

    def __init__(self, db: Database, data_dir: Path, dataset_id: str = None):
        super().__init__()
        self.db = db
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.plotter = MapPlotter(data_dir)
        self.exporter = DataExporter(data_dir)

        self._dataset: Optional[Dataset] = None
        self._sites: list[Site] = []
        self._current_hour = 12
        self._is_animating = False

        self._build()

    def did_mount(self):
        """Called when control is added to page - load data async."""
        self.page.run_task(self._load_data_async)

    async def _load_data_async(self):
        """Load dataset and related data."""
        if self.dataset_id:
            self._dataset = await asyncio.to_thread(self.db.get_dataset, self.dataset_id)
            if self._dataset:
                self._dataset_title.value = self._dataset.name
                self._sites = await asyncio.to_thread(
                    self.db.get_sites_in_bbox, self._dataset.bbox
                )
                self._update_sites_list()
                self._status_text.value = f"Loaded: {self._dataset.name}"
        self.update()

    def _build(self):
        """Build the workspace layout."""
        # Header with dataset name
        self._dataset_title = ft.Text(
            "Loading...",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=Colors.ON_SURFACE,
        )

        self._status_text = ft.Text(
            "",
            size=12,
            color=Colors.ON_SURFACE_VARIANT,
        )

        header = ft.Container(
            content=ft.Row([
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    icon_color=Colors.ON_SURFACE,
                    tooltip="Back to Library",
                    on_click=self._on_back_click,
                ),
                ft.Column([
                    self._dataset_title,
                    self._status_text,
                ], spacing=2),
            ], spacing=8),
            padding=ft.padding.only(bottom=Spacing.MD),
        )

        # Create tabs
        self._tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(text="Plot", icon=ft.Icons.MAP),
                ft.Tab(text="Export", icon=ft.Icons.FILE_DOWNLOAD),
                ft.Tab(text="Sites", icon=ft.Icons.LOCATION_ON),
            ],
            on_change=self._on_tab_change,
        )

        # Tab content containers
        self._plot_content = self._build_plot_tab()
        self._export_content = self._build_export_tab()
        self._sites_content = self._build_sites_tab()

        self._tab_content = ft.Container(
            content=self._plot_content,
            expand=True,
        )

        # Main layout
        self.content = ft.Column([
            header,
            self._tabs,
            self._tab_content,
        ], expand=True)
        self.expand = True
        self.padding = Spacing.PAGE_HORIZONTAL

    def _build_plot_tab(self):
        """Build the plot tab content."""
        # Variable selector
        self._variable_dropdown = ft.Dropdown(
            label="Variable",
            value="NO2_TropVCD",
            options=[
                ft.DropdownOption(key="NO2_TropVCD", text="NO2 Tropospheric VCD"),
                ft.DropdownOption(key="HCHO_TotVCD", text="HCHO Total VCD"),
                ft.DropdownOption(key="FNR", text="FNR (HCHO/NO2)"),
            ],
            width=250,
            border_color=Colors.BORDER,
        )

        # Hour slider
        self._hour_slider = ft.Slider(
            min=0, max=23, divisions=23, value=12,
            label="{value}",
            on_change=self._on_hour_change,
        )
        self._hour_text = ft.Text("Hour: 12 UTC", size=13)

        # Road options
        self._road_dropdown = ft.Dropdown(
            label="Roads",
            value="primary",
            options=[
                ft.DropdownOption(key="primary", text="Interstates Only"),
                ft.DropdownOption(key="major", text="Major Roads"),
                ft.DropdownOption(key="all", text="All Roads"),
            ],
            width=150,
            border_color=Colors.BORDER,
        )

        self._show_sites_checkbox = ft.Checkbox(
            label="Show Sites",
            value=True,
        )

        # Generate button
        self._generate_btn = ft.FilledButton(
            content=ft.Row([
                ft.Icon(ft.Icons.MAP, size=20),
                ft.Text("Generate Map"),
            ], spacing=8, tight=True),
            on_click=self._on_generate_click,
        )

        # Map image display
        self._map_image = ft.Image(
            src="",
            fit=ft.ImageFit.CONTAIN,
            visible=False,
        )

        self._map_placeholder = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.MAP, size=64, color=Colors.ON_SURFACE_VARIANT),
                ft.Text("Select options and click Generate Map", color=Colors.ON_SURFACE_VARIANT),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=16),
            alignment=ft.Alignment(0, 0),
            expand=True,
            bgcolor=Colors.SURFACE_VARIANT,
            border_radius=8,
        )

        self._progress_bar = ft.ProgressBar(visible=False, color=Colors.PRIMARY)
        self._plot_status = ft.Text("", size=12, color=Colors.ON_SURFACE_VARIANT)

        return ft.Column([
            ft.Row([
                self._variable_dropdown,
                ft.Container(width=16),
                self._road_dropdown,
                ft.Container(width=16),
                self._show_sites_checkbox,
                ft.Container(expand=True),
                self._generate_btn,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(height=8),
            ft.Row([
                self._hour_text,
                ft.Container(content=self._hour_slider, expand=True),
            ]),
            self._progress_bar,
            self._plot_status,
            ft.Container(height=8),
            ft.Container(
                content=ft.Stack([
                    self._map_placeholder,
                    self._map_image,
                ]),
                expand=True,
                border=ft.border.all(1, Colors.BORDER),
                border_radius=8,
            ),
        ], expand=True)

    def _build_export_tab(self):
        """Build the export tab content."""
        # Export format
        self._export_format = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="hourly", label="Hourly (per-site files)"),
                ft.Radio(value="daily", label="Daily (merged file)"),
            ]),
            value="hourly",
        )

        # Num points
        self._num_points_dropdown = ft.Dropdown(
            label="Grid Cells",
            value="4",
            options=[
                ft.DropdownOption(key="4", text="4 cells"),
                ft.DropdownOption(key="8", text="8 cells"),
                ft.DropdownOption(key="9", text="9 cells"),
            ],
            width=120,
        )

        # UTC Offset
        self._utc_offset_field = ft.TextField(
            label="UTC Offset",
            value="-6.0",
            width=100,
            text_align=ft.TextAlign.CENTER,
        )

        # Export button
        self._export_btn = ft.FilledButton(
            content=ft.Row([
                ft.Icon(ft.Icons.FILE_DOWNLOAD, size=20),
                ft.Text("Export to Excel"),
            ], spacing=8, tight=True),
            on_click=self._on_export_click,
        )

        # Status log
        self._export_log = StatusLogPanel()

        return ft.Column([
            SectionCard(
                title="Export Settings",
                icon=ft.Icons.SETTINGS,
                content=ft.Column([
                    ft.Text("Format", size=13, weight=ft.FontWeight.W_600),
                    self._export_format,
                    ft.Container(height=8),
                    ft.Row([
                        self._num_points_dropdown,
                        ft.Container(width=16),
                        self._utc_offset_field,
                    ]),
                ], spacing=8),
            ),
            ft.Container(height=16),
            ft.Row([
                self._export_btn,
            ]),
            ft.Container(height=16),
            ft.Container(
                content=self._export_log,
                expand=True,
            ),
        ], expand=True)

    def _build_sites_tab(self):
        """Build the sites tab content."""
        self._sites_list = ft.ListView(spacing=8, expand=True)

        self._sites_count = ft.Text(
            "0 sites in dataset bounds",
            size=13,
            color=Colors.ON_SURFACE_VARIANT,
        )

        return ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.LOCATION_ON, size=20, color=Colors.PRIMARY),
                ft.Text("Sites in Dataset", size=16, weight=ft.FontWeight.W_600),
                ft.Container(expand=True),
                self._sites_count,
            ]),
            ft.Container(height=8),
            self._sites_list,
        ], expand=True)

    def _update_sites_list(self):
        """Update the sites list display."""
        self._sites_list.controls.clear()
        self._sites_count.value = f"{len(self._sites)} sites in dataset bounds"

        for site in self._sites:
            self._sites_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PLACE, size=18, color=Colors.INFO),
                        ft.Column([
                            ft.Text(site.code, weight=ft.FontWeight.W_600, size=14),
                            ft.Text(
                                f"{site.latitude:.4f}, {site.longitude:.4f}",
                                size=12,
                                color=Colors.ON_SURFACE_VARIANT,
                            ),
                        ], spacing=2, expand=True),
                    ], spacing=12),
                    padding=12,
                    bgcolor=Colors.SURFACE,
                    border_radius=8,
                    border=ft.border.all(1, Colors.BORDER),
                )
            )

    def _on_tab_change(self, e):
        """Handle tab selection change."""
        idx = e.control.selected_index
        if idx == 0:
            self._tab_content.content = self._plot_content
        elif idx == 1:
            self._tab_content.content = self._export_content
        elif idx == 2:
            self._tab_content.content = self._sites_content
        self.update()

    def _on_back_click(self, e):
        """Navigate back to library."""
        if self.page:
            shell = self.page.controls[0] if self.page.controls else None
            if shell and hasattr(shell, 'navigate_to'):
                shell.navigate_to("/library")

    def _on_hour_change(self, e):
        """Handle hour slider change."""
        hour = int(e.control.value)
        self._current_hour = hour
        self._hour_text.value = f"Hour: {hour} UTC"
        self.update()

    def _on_generate_click(self, e):
        """Generate the map."""
        if not self._dataset:
            return
        self.page.run_task(self._generate_map_async)

    async def _generate_map_async(self):
        """Generate map asynchronously."""
        self._progress_bar.visible = True
        self._plot_status.value = "Generating map..."
        self.update()

        try:
            # Find processed file
            if self._dataset.file_path and Path(self._dataset.file_path).exists():
                processed_path = Path(self._dataset.file_path)
            else:
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in self._dataset.name)
                processed_path = self.data_dir / "datasets" / safe_name / f"{safe_name}_processed.nc"

            if not processed_path.exists():
                self._plot_status.value = "Processed data not found"
                self._progress_bar.visible = False
                self.update()
                return

            ds = await asyncio.to_thread(xr.open_dataset, processed_path)

            # Get sites if checkbox checked
            sites = None
            if self._show_sites_checkbox.value:
                sites = {s.code: s.to_tuple() for s in self._sites}

            variable = self._variable_dropdown.value
            road_detail = self._road_dropdown.value
            hour = self._current_hour

            plot_path = await asyncio.to_thread(
                self.plotter.generate_map,
                ds,
                hour,
                variable,
                self._dataset.name,
                self._dataset.bbox.to_list(),
                road_detail,
                sites,
            )

            ds.close()

            if plot_path and Path(plot_path).exists():
                self._map_image.src = plot_path
                self._map_image.visible = True
                self._map_placeholder.visible = False
                self._plot_status.value = f"Map generated: {variable} at {hour}:00 UTC"
            else:
                self._plot_status.value = "Failed to generate map"

        except Exception as ex:
            self._plot_status.value = f"Error: {ex}"
        finally:
            self._progress_bar.visible = False
            self.update()

    def _on_export_click(self, e):
        """Export data to Excel."""
        if not self._dataset:
            self._export_log.add_error("No dataset selected")
            return
        self.page.run_task(self._export_async)

    async def _export_async(self):
        """Export data asynchronously."""
        self._export_log.add_info("Starting export...")

        try:
            # Find processed file
            if self._dataset.file_path and Path(self._dataset.file_path).exists():
                processed_path = Path(self._dataset.file_path)
            else:
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in self._dataset.name)
                processed_path = self.data_dir / "datasets" / safe_name / f"{safe_name}_processed.nc"

            if not processed_path.exists():
                self._export_log.add_error("Processed data not found")
                return

            ds = await asyncio.to_thread(xr.open_dataset, processed_path)

            export_format = self._export_format.value
            num_points = int(self._num_points_dropdown.value)
            utc_offset = float(self._utc_offset_field.value)

            # Get sites from database
            sites = {s.code: s.to_tuple() for s in self._sites}

            metadata = {
                'dataset_name': self._dataset.name,
                'max_cloud': self._dataset.max_cloud,
                'max_sza': self._dataset.max_sza,
            }

            generated_files = await asyncio.to_thread(
                self.exporter.export_dataset,
                ds,
                self._dataset.name,
                export_format,
                num_points,
                utc_offset,
                metadata,
                sites,
            )

            ds.close()

            if generated_files:
                self._export_log.add_success(f"Export complete! Generated {len(generated_files)} file(s):")
                for fpath in generated_files:
                    self._export_log.add_success(f"  {fpath}")
            else:
                self._export_log.add_warning("No files generated. Check if sites exist in dataset bounds.")

        except Exception as ex:
            self._export_log.add_error(f"Export failed: {ex}")
