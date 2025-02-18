from typing import Dict, Any, List, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from scipy import stats
import logging
from pathlib import Path

class DataPlotter:
    """Visualization utilities for scientific data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_theme = self.config.get('theme', 'plotly')
        self.default_colors = px.colors.qualitative.Set1
        
    def create_scatter_plot(self,
                          data: Union[pd.DataFrame, np.ndarray],
                          x: Optional[str] = None,
                          y: Optional[str] = None,
                          color: Optional[str] = None,
                          size: Optional[str] = None,
                          title: str = "Scatter Plot",
                          **kwargs) -> Dict[str, Any]:
        """Create interactive scatter plot"""
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'dim_{i}' for i in range(data.shape[1])])
                x = x or 'dim_0'
                y = y or 'dim_1'
            
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color=color,
                size=size,
                title=title,
                template=self.default_theme,
                **kwargs
            )
            
            return {
                'plotly_json': fig.to_json(),
                'data_summary': {
                    'n_points': len(data),
                    'x_range': [float(data[x].min()), float(data[x].max())],
                    'y_range': [float(data[y].min()), float(data[y].max())]
                }
            }
            
        except Exception as e:
            logging.error(f"Scatter plot creation error: {str(e)}")
            raise
            
    def create_line_plot(self,
                        data: Union[pd.DataFrame, np.ndarray],
                        x: Optional[str] = None,
                        y: Union[str, List[str]] = None,
                        title: str = "Line Plot",
                        **kwargs) -> Dict[str, Any]:
        """Create interactive line plot"""
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'series_{i}' for i in range(data.shape[1])])
                x = x or data.index
                y = y or 'series_0'
            
            fig = px.line(
                data,
                x=x,
                y=y,
                title=title,
                template=self.default_theme,
                **kwargs
            )
            
            return {
                'plotly_json': fig.to_json(),
                'data_summary': {
                    'n_points': len(data),
                    'y_range': [float(data[y].min()), float(data[y].max())] if isinstance(y, str)
                              else [float(data[y_col].min()) for y_col in y]
                }
            }
            
        except Exception as e:
            logging.error(f"Line plot creation error: {str(e)}")
            raise
            
    def create_heatmap(self,
                      data: Union[pd.DataFrame, np.ndarray],
                      title: str = "Heatmap",
                      colorscale: str = "RdBu",
                      **kwargs) -> Dict[str, Any]:
        """Create interactive heatmap"""
        try:
            if isinstance(data, pd.DataFrame):
                matrix = data.values
                labels = {'x': data.columns, 'y': data.index}
            else:
                matrix = data
                labels = {'x': None, 'y': None}
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                colorscale=colorscale,
                **kwargs
            ))
            
            fig.update_layout(
                title=title,
                template=self.default_theme
            )
            
            return {
                'plotly_json': fig.to_json(),
                'data_summary': {
                    'shape': matrix.shape,
                    'value_range': [float(np.min(matrix)), float(np.max(matrix))]
                }
            }
            
        except Exception as e:
            logging.error(f"Heatmap creation error: {str(e)}")
            raise
            
    def create_distribution_plot(self,
                               data: Union[pd.DataFrame, np.ndarray],
                               columns: Optional[List[str]] = None,
                               plot_type: str = 'histogram',
                               title: str = "Distribution Plot",
                               **kwargs) -> Dict[str, Any]:
        """Create distribution visualization"""
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
            
            cols_to_plot = columns or data.select_dtypes(include=[np.number]).columns
            
            if plot_type == 'histogram':
                fig = px.histogram(
                    data,
                    x=cols_to_plot,
                    title=title,
                    template=self.default_theme,
                    **kwargs
                )
            elif plot_type == 'box':
                fig = px.box(
                    data,
                    y=cols_to_plot,
                    title=title,
                    template=self.default_theme,
                    **kwargs
                )
            elif plot_type == 'violin':
                fig = px.violin(
                    data,
                    y=cols_to_plot,
                    title=title,
                    template=self.default_theme,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            # Calculate distribution statistics
            stats_dict = {}
            for col in cols_to_plot:
                col_data = data[col].dropna()
                stats_dict[col] = {
                    'mean': float(np.mean(col_data)),
                    'std': float(np.std(col_data)),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data))
                }
            
            return {
                'plotly_json': fig.to_json(),
                'distribution_stats': stats_dict
            }
            
        except Exception as e:
            logging.error(f"Distribution plot creation error: {str(e)}")
            raise
    
    def create_3d_scatter(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         x: Optional[str] = None,
                         y: Optional[str] = None,
                         z: Optional[str] = None,
                         color: Optional[str] = None,
                         title: str = "3D Scatter Plot",
                         **kwargs) -> Dict[str, Any]:
        """Create interactive 3D scatter plot"""
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(
                    data,
                    columns=['x', 'y', 'z'] if data.shape[1] == 3 else [f'dim_{i}' for i in range(data.shape[1])]
                )
                x = x or 'x'
                y = y or 'y'
                z = z or 'z'
            
            fig = px.scatter_3d(
                data,
                x=x,
                y=y,
                z=z,
                color=color,
                title=title,
                template=self.default_theme,
                **kwargs
            )
            
            return {
                'plotly_json': fig.to_json(),
                'data_summary': {
                    'n_points': len(data),
                    'x_range': [float(data[x].min()), float(data[x].max())],
                    'y_range': [float(data[y].min()), float(data[y].max())],
                    'z_range': [float(data[z].min()), float(data[z].max())]
                }
            }
            
        except Exception as e:
            logging.error(f"3D scatter plot creation error: {str(e)}")
            raise
