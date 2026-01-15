"""Application shell with navigation rail and page routing."""

import flet as ft
from typing import Callable

from .theme import Colors, Spacing


class NavigationItem:
    """Navigation rail item configuration."""
    
    def __init__(
        self,
        icon: str,
        selected_icon: str,
        label: str,
        route: str,
    ):
        self.icon = icon
        self.selected_icon = selected_icon
        self.label = label
        self.route = route


# Navigation items for the app - organized by workflow
NAV_ITEMS = [
    # === DATA ===
    NavigationItem(
        icon=ft.Icons.ADD_CHART_OUTLINED,
        selected_icon=ft.Icons.ADD_CHART,
        label="New",
        route="/create",
    ),
    NavigationItem(
        icon=ft.Icons.FOLDER_OUTLINED,
        selected_icon=ft.Icons.FOLDER,
        label="Datasets",
        route="/library",
    ),
    NavigationItem(
        icon=ft.Icons.UPLOAD_FILE_OUTLINED,
        selected_icon=ft.Icons.UPLOAD_FILE,
        label="Batch",
        route="/batch",
    ),
    # === ANALYSIS ===
    NavigationItem(
        icon=ft.Icons.MAP_OUTLINED,
        selected_icon=ft.Icons.MAP,
        label="Maps",
        route="/plot",
    ),
    NavigationItem(
        icon=ft.Icons.INSIGHTS_OUTLINED,
        selected_icon=ft.Icons.INSIGHTS,
        label="Explore",
        route="/inspect",
    ),
    # === EXPORT ===
    NavigationItem(
        icon=ft.Icons.FILE_DOWNLOAD_OUTLINED,
        selected_icon=ft.Icons.FILE_DOWNLOAD,
        label="Export",
        route="/export",
    ),
    # === CONFIG ===
    NavigationItem(
        icon=ft.Icons.PLACE_OUTLINED,
        selected_icon=ft.Icons.PLACE,
        label="Sites",
        route="/sites",
    ),
    NavigationItem(
        icon=ft.Icons.SETTINGS_OUTLINED,
        selected_icon=ft.Icons.SETTINGS,
        label="Settings",
        route="/settings",
    ),
]


class AppShell(ft.Container):
    """Main application shell with navigation and content area."""
    
    def __init__(
        self,
        page: ft.Page,
        on_route_change: Callable[[str], None] = None,
    ):
        super().__init__()
        self._page = page  # Use _page to avoid conflict with Container.page
        self._on_route_change_callback = on_route_change
        self._selected_index = 0
        self._selected_index = 0
        
        # Content placeholder
        self._content_area = ft.Container(
            expand=True,
            padding=Spacing.PAGE_HORIZONTAL,
        )
        
        # Build the shell
        self._build()
    
    def _build(self):
        """Build the shell layout."""
        # Navigation rail
        self._nav_rail = ft.NavigationRail(
            selected_index=self._selected_index,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=Spacing.NAV_RAIL_WIDTH,
            min_extended_width=Spacing.NAV_RAIL_EXPANDED,
            bgcolor=Colors.SURFACE,
            indicator_color=Colors.PRIMARY_CONTAINER,
            destinations=[
                ft.NavigationRailDestination(
                    icon=item.icon,
                    selected_icon=item.selected_icon,
                    label=item.label,
                )
                for item in NAV_ITEMS
            ],
            on_change=self._on_nav_change,
        )
        
        # App bar
        self._app_bar = ft.Container(
            content=ft.Row(
                controls=[
                    # Logo and title
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.SATELLITE_ALT, color=Colors.PRIMARY, size=28),
                            ft.Text(
                                "TEMPO Analyzer",
                                size=20,
                                weight=ft.FontWeight.W_600,
                                color=Colors.ON_SURFACE,
                            ),
                        ],
                        spacing=Spacing.SM,
                    ),
                    # Spacer
                    ft.Container(expand=True),
                    # Settings button
                    ft.IconButton(
                        icon=ft.Icons.SETTINGS_OUTLINED,
                        icon_color=Colors.ON_SURFACE_VARIANT,
                        tooltip="Settings",
                        on_click=lambda _: self._navigate_to("/settings"),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.symmetric(horizontal=Spacing.LG, vertical=Spacing.SM),
            bgcolor=Colors.SURFACE,
            border=ft.border.only(bottom=ft.BorderSide(1, Colors.BORDER)),
        )
        
        # Main layout
        self.content = ft.Column(
            controls=[
                # App bar
                self._app_bar,
                # Main content area with nav rail
                ft.Row(
                    controls=[
                        # Navigation rail
                        ft.Container(
                            content=self._nav_rail,
                            border=ft.border.only(right=ft.BorderSide(1, Colors.BORDER)),
                        ),
                        # Content area
                        self._content_area,
                    ],
                    expand=True,
                    spacing=0,
                ),
            ],
            spacing=0,
            expand=True,
        )
        
        self.bgcolor = Colors.BACKGROUND
        self.expand = True
    
    def _on_nav_change(self, e: ft.ControlEvent):
        """Handle navigation rail selection change."""
        self._selected_index = e.control.selected_index
        route = NAV_ITEMS[self._selected_index].route
        self._navigate_to(route)
    
    def _navigate_to(self, route: str):
        """Navigate to a route."""
        # Update nav rail selection
        for i, item in enumerate(NAV_ITEMS):
            if item.route == route:
                self._selected_index = i
                self._nav_rail.selected_index = i
                break
        
        # Notify callback
        if self._on_route_change_callback:
            self._on_route_change_callback(route)
        
        self._page.update()
    
        self._page.update()
    
    def set_content(self, content: ft.Control):
        """Set the main content area."""
        self._content_area.content = content
        if self._page:
            self._page.update()
    
    def navigate_to(self, route: str):
        """Public method to navigate to a route."""
        self._navigate_to(route)
    
    @property
    def selected_route(self) -> str:
        """Get the currently selected route."""
        return NAV_ITEMS[self._selected_index].route
