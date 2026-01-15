"""Sites Management Page - Add, view, and delete monitoring sites."""

import flet as ft
from datetime import datetime
from typing import Optional

from ..theme import Colors, Spacing
from ..components.widgets import SectionCard
from ...storage.database import Database
from ...storage.models import Site


class SitesPage(ft.Container):
    """Page for managing monitoring sites that appear on maps."""
    
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._build()
    
    def _build(self):
        """Build the sites management page."""
        # Header
        header = ft.Row([
            ft.Icon(ft.Icons.PLACE, size=28, color=Colors.PRIMARY),
            ft.Text("Site Management", size=24, weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE),
        ], spacing=12)
        
        description = ft.Text(
            "Manage monitoring sites that are marked on map visualizations. "
            "Sites are shown as star markers with their code labels.",
            size=13,
            color=Colors.ON_SURFACE_VARIANT,
        )
        
        # === Add Site Form ===
        self._code_field = ft.TextField(
            label="Site Code",
            hint_text="e.g., BV, LC",
            width=100,
            border_color=Colors.BORDER,
            bgcolor=Colors.SURFACE_VARIANT,
            max_length=10,
            text_style=ft.TextStyle(color=Colors.ON_SURFACE),
            label_style=ft.TextStyle(color=Colors.ON_SURFACE),
        )
        
        self._name_field = ft.TextField(
            label="Site Name (optional)",
            hint_text="e.g., Bountiful, UT",
            width=200,
            border_color=Colors.BORDER,
            bgcolor=Colors.SURFACE_VARIANT,
            text_style=ft.TextStyle(color=Colors.ON_SURFACE),
            label_style=ft.TextStyle(color=Colors.ON_SURFACE),
        )
        
        self._lat_field = ft.TextField(
            label="Latitude",
            hint_text="e.g., 40.903",
            width=120,
            border_color=Colors.BORDER,
            bgcolor=Colors.SURFACE_VARIANT,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_style=ft.TextStyle(color=Colors.ON_SURFACE),
            label_style=ft.TextStyle(color=Colors.ON_SURFACE),
        )
        
        self._lon_field = ft.TextField(
            label="Longitude",
            hint_text="e.g., -111.884",
            width=120,
            border_color=Colors.BORDER,
            bgcolor=Colors.SURFACE_VARIANT,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_style=ft.TextStyle(color=Colors.ON_SURFACE),
            label_style=ft.TextStyle(color=Colors.ON_SURFACE),
        )
        
        self._add_btn = ft.ElevatedButton(
            "Add Site",
            icon=ft.Icons.ADD_LOCATION,
            bgcolor=Colors.PRIMARY,
            color=Colors.ON_PRIMARY,
            on_click=self._on_add_site,
        )
        
        self._status_text = ft.Text("", size=12, color=Colors.ON_SURFACE_VARIANT)
        
        add_form = ft.Container(
            content=ft.Column([
                ft.Text("Add New Site", size=16, weight=ft.FontWeight.W_600, color=Colors.ON_SURFACE),
                ft.Container(height=8),
                ft.Row([
                    self._code_field,
                    self._name_field,
                    self._lat_field,
                    self._lon_field,
                    self._add_btn,
                ], spacing=12, wrap=True),
                self._status_text,
            ]),
            bgcolor=Colors.SURFACE,
            padding=16,
            border_radius=8,
            border=ft.border.all(1, Colors.BORDER),
        )
        
        # === Import Defaults Button ===
        self._import_btn = ft.OutlinedButton(
            "Import Default Sites",
            icon=ft.Icons.DOWNLOAD,
            on_click=self._on_import_defaults,
        )
        
        # === Sites List ===
        self._sites_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Code", weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE)),
                ft.DataColumn(ft.Text("Latitude", weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE), numeric=True),
                ft.DataColumn(ft.Text("Longitude", weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE), numeric=True),
                ft.DataColumn(ft.Text("Actions", weight=ft.FontWeight.BOLD, color=Colors.ON_SURFACE)),
            ],
            rows=[],
            border=ft.border.all(1, Colors.BORDER),
            border_radius=8,
            heading_row_color=Colors.SURFACE_VARIANT,
            data_row_max_height=48,
        )
        
        sites_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Registered Sites", size=16, weight=ft.FontWeight.W_600, color=Colors.ON_SURFACE),
                    ft.Container(expand=True),
                    self._import_btn,
                    ft.IconButton(
                        icon=ft.Icons.REFRESH,
                        tooltip="Refresh list",
                        on_click=self._on_refresh,
                    ),
                ]),
                ft.Container(height=8),
                self._sites_table,
            ]),
            bgcolor=Colors.SURFACE,
            padding=16,
            border_radius=8,
            border=ft.border.all(1, Colors.BORDER),
        )
        
        # === Main Layout ===
        self.content = ft.Column([
            header,
            description,
            ft.Container(height=16),
            add_form,
            ft.Container(height=16),
            sites_card,
        ], scroll=ft.ScrollMode.AUTO)
        
        self.expand = True
        self.padding = Spacing.PAGE_HORIZONTAL
        
        # Load initial data
        self._load_sites()
    
    def _load_sites(self):
        """Load sites from database into table."""
        sites = self.db.get_all_sites()
        
        rows = []
        for site in sites:
            rows.append(ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(site.code, weight=ft.FontWeight.W_600, color=Colors.ON_SURFACE)),
                    ft.DataCell(ft.Text(site.name or "-", color=Colors.ON_SURFACE)),
                    ft.DataCell(ft.Text(f"{site.latitude:.4f}", color=Colors.ON_SURFACE)),
                    ft.DataCell(ft.Text(f"{site.longitude:.4f}", color=Colors.ON_SURFACE)),
                    ft.DataCell(
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=Colors.ERROR,
                            tooltip="Delete site",
                            data=site.id,  # Store ID for deletion
                            on_click=self._on_delete_site,
                        )
                    ),
                ]
            ))
        
        self._sites_table.rows = rows
        
        if not sites:
            self._status_text.value = "No sites registered. Add sites above or import defaults."
            self._status_text.color = Colors.ON_SURFACE_VARIANT
        else:
            self._status_text.value = f"{len(sites)} site(s) registered"
            self._status_text.color = Colors.ON_SURFACE_VARIANT
    
    def _on_add_site(self, e):
        """Handle adding a new site."""
        code = self._code_field.value.strip().upper()
        name = self._name_field.value.strip()
        lat_str = self._lat_field.value.strip()
        lon_str = self._lon_field.value.strip()
        
        # Validation
        if not code:
            self._show_status("Error: Site code is required", error=True)
            return
        
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            self._show_status("Error: Invalid latitude or longitude", error=True)
            return
        
        if not (-90 <= lat <= 90):
            self._show_status("Error: Latitude must be between -90 and 90", error=True)
            return
        
        if not (-180 <= lon <= 180):
            self._show_status("Error: Longitude must be between -180 and 180", error=True)
            return
        
        # Create site
        try:
            site = Site(
                code=code,
                name=name,
                latitude=lat,
                longitude=lon,
                created_at=datetime.now(),
            )
            self.db.create_site(site)
            
            # Clear form
            self._code_field.value = ""
            self._name_field.value = ""
            self._lat_field.value = ""
            self._lon_field.value = ""
            
            self._show_status(f"✓ Site '{code}' added successfully")
            self._load_sites()
            self.update()
            
        except Exception as ex:
            if "UNIQUE constraint" in str(ex):
                self._show_status(f"Error: Site code '{code}' already exists", error=True)
            else:
                self._show_status(f"Error: {ex}", error=True)
    
    def _on_delete_site(self, e):
        """Handle deleting a site."""
        site_id = e.control.data
        if site_id:
            self.db.delete_site(site_id)
            self._show_status("Site deleted")
            self._load_sites()
            self.update()
    
    def _on_import_defaults(self, e):
        """Import default hardcoded sites."""
        added = self.db.seed_default_sites()
        if added > 0:
            self._show_status(f"✓ Imported {added} default site(s)")
        else:
            self._show_status("No new sites to import (all defaults already exist)")
        self._load_sites()
        self.update()
    
    def _on_refresh(self, e):
        """Refresh the sites list."""
        self._load_sites()
        self.update()
    
    def _show_status(self, message: str, error: bool = False):
        """Show a status message."""
        self._status_text.value = message
        self._status_text.color = Colors.ERROR if error else Colors.SUCCESS
