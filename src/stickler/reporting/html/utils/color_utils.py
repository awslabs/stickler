"""
Color utilities for HTML reporting.
Centralizes color logic that was previously duplicated across multiple modules.
"""

from typing import Dict


class ColorUtils:
    """Centralized color utilities for performance indicators and themes."""
    
    # Performance score color thresholds
    PERFORMANCE_THRESHOLDS = {
        'EXCELLENT': 0.8,
        'GOOD': 0.6,
        'FAIR': 0.4
    }
    
    # Default color palette
    DEFAULT_COLORS = {
        'GREEN': '#28a745',
        'YELLOW': '#ffc107',
        'ORANGE': '#fd7e14',
        'RED': '#dc3545',
        'BLUE': '#007bff',
        'GRAY': '#6c757d'
    }

    
    @staticmethod
    def get_performance_color(score: float, thresholds: Dict = None) -> str:
        """
        Get color based on performance score.
        
        Args:
            score: Performance score (0.0 to 1.0)
            
        Returns:
            Hex color code string
        """
        if thresholds is None:
            thresholds = ColorUtils.PERFORMANCE_THRESHOLDS
        
        if score >= thresholds.get('EXCELLENT', 0.8):
            return ColorUtils.DEFAULT_COLORS['GREEN']
        elif score >= thresholds.get('GOOD', 0.6):
            return ColorUtils.DEFAULT_COLORS['YELLOW']
        elif score >= thresholds.get('FAIR', 0.4):
            return ColorUtils.DEFAULT_COLORS['ORANGE']
        else:
            return ColorUtils.DEFAULT_COLORS['RED']
    
    @staticmethod
    def get_status_color(status: str) -> str:
        """
        Get color for status indicators.
        
        Args:
            status: Status type ('pass', 'fail', 'warning', 'info')
            theme: Theme name (unused in simplified version)
            
        Returns:
            Hex color code string
        """
        status_mapping = {
            'pass': ColorUtils.DEFAULT_COLORS['GREEN'],
            'fail': ColorUtils.DEFAULT_COLORS['RED'],
            'warning': ColorUtils.DEFAULT_COLORS['YELLOW'],
            'info': ColorUtils.DEFAULT_COLORS['BLUE'],
            'error': ColorUtils.DEFAULT_COLORS['RED'],
            'success': ColorUtils.DEFAULT_COLORS['GREEN']
        }
        
        return status_mapping.get(status.lower(), ColorUtils.DEFAULT_COLORS['BLUE'])
    
    @staticmethod
    def get_confusion_matrix_colors() -> Dict[str, str]:
        """
        Get color mapping for confusion matrix elements.
        
        Returns:
            Dictionary mapping confusion matrix elements to colors
        """
        return {
            'tp': ColorUtils.DEFAULT_COLORS['GREEN'],   
            'tn': ColorUtils.DEFAULT_COLORS['GREEN'],    
            'fp': ColorUtils.DEFAULT_COLORS['RED'],     
            'fa': ColorUtils.DEFAULT_COLORS['RED'],     
            'fn': ColorUtils.DEFAULT_COLORS['RED'],     
            'fd': ColorUtils.DEFAULT_COLORS['RED']      
        }
