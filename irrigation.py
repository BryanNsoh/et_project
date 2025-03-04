"""
Irrigation calculations based on ET data.
"""
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

def compute_daily_replacement(
    df: pd.DataFrame,
    et_col: str = "ET",
    rain_col: Optional[str] = None,
    efficiency: float = 1.0
) -> pd.DataFrame:
    """
    Compute daily irrigation needs using the replacement method.
    Each day's irrigation = ET - effective rainfall.
    
    Args:
        df: DataFrame with ET data and optionally rainfall
        et_col: Column name for ET values
        rain_col: Column name for rainfall (optional)
        efficiency: Irrigation system efficiency (1.0 = 100%)
        
    Returns:
        pd.DataFrame: Original dataframe with irrigation needs column added
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate net ET (ET minus rainfall)
    if rain_col and rain_col in df.columns:
        result_df["Net_ET"] = (df[et_col] - df[rain_col]).clip(lower=0)
    else:
        result_df["Net_ET"] = df[et_col]
    
    # Calculate irrigation need considering efficiency
    if efficiency < 1.0:
        result_df["Irrigation_mm"] = result_df["Net_ET"] / efficiency
    else:
        result_df["Irrigation_mm"] = result_df["Net_ET"]
    
    return result_df

def compute_threshold_schedule(
    df: pd.DataFrame,
    threshold_mm: float,
    et_col: str = "ET",
    rain_col: Optional[str] = None,
    efficiency: float = 1.0,
) -> List[Tuple[datetime, float]]:
    """
    Compute an irrigation schedule based on a soil water depletion threshold.
    When cumulative deficit reaches threshold, schedule an irrigation.
    
    Args:
        df: DataFrame with ET data (must have 'date' column)
        threshold_mm: Threshold in mm to trigger irrigation
        et_col: Column name for ET values
        rain_col: Column name for rainfall (optional)
        efficiency: Irrigation system efficiency (1.0 = 100%)
        
    Returns:
        List[Tuple[datetime, float]]: List of (date, amount) tuples for irrigation events
    """
    if threshold_mm <= 0:
        raise ValueError("Threshold must be positive")
    
    # Calculate net ET
    if rain_col and rain_col in df.columns:
        net_et = (df[et_col] - df[rain_col]).clip(lower=0)
    else:
        net_et = df[et_col]
    
    # Compute schedule
    deficit = 0
    schedule = []
    
    for i, row in df.iterrows():
        date = row['date']
        daily_net_et = net_et.iloc[i]
        
        # Add to deficit
        deficit += daily_net_et
        
        # Check if threshold reached or last day
        if deficit >= threshold_mm or i == len(df) - 1:
            if deficit > 0:  # Only schedule if deficit exists
                # Account for efficiency
                irrigation_amount = deficit / efficiency if efficiency < 1.0 else deficit
                schedule.append((date, irrigation_amount))
                deficit = 0  # Reset deficit after irrigation
    
    return schedule

def get_irrigation_recommendation(
    df: pd.DataFrame,
    mode: str = "daily",
    threshold_mm: float = 25.0,
    system_efficiency: float = 0.85,
    et_col: str = "ET",
    rain_col: Optional[str] = None,
    units: str = "mm",
) -> Dict:
    """
    Generate irrigation recommendations from ET data.
    
    Args:
        df: DataFrame with ET and optional rain data
        mode: Recommendation mode ('daily' or 'threshold')
        threshold_mm: Depletion threshold for threshold mode
        system_efficiency: Irrigation system efficiency (0.0-1.0)
        et_col: Column name for ET
        rain_col: Column name for rainfall
        units: Output units ('mm' or 'inches')
    
    Returns:
        Dict: Results and recommendations
    """
    # Validate inputs
    if mode not in ["daily", "threshold"]:
        raise ValueError("Mode must be 'daily' or 'threshold'")
    
    if not 0 < system_efficiency <= 1.0:
        raise ValueError("System efficiency must be between 0 and 1")
    
    # Check if we have data
    if len(df) == 0:
        return {
            "status": "error",
            "message": "No data provided for irrigation calculations",
            "recommendation": "Unable to generate recommendations without data"
        }
    
    # Prepare summary statistics
    total_et = df[et_col].sum()
    total_rain = df[rain_col].sum() if rain_col and rain_col in df.columns else 0
    period_days = (df['date'].max() - df['date'].min()).days + 1
    
    # Generate recommendations based on mode
    if mode == "daily":
        # Daily replacement mode
        result_df = compute_daily_replacement(
            df, 
            et_col=et_col, 
            rain_col=rain_col, 
            efficiency=system_efficiency
        )
        
        # Total irrigation needed
        total_irrigation = result_df["Irrigation_mm"].sum()
        
        # Convert to specified units if needed
        conversion_factor = 1.0 if units == "mm" else 1/25.4  # mm to inches
        
        total_et_display = total_et * conversion_factor
        total_rain_display = total_rain * conversion_factor
        total_irrigation_display = total_irrigation * conversion_factor
        display_unit = "mm" if units == "mm" else "inches"
        
        # Build recommendation text
        if total_irrigation <= 0:
            recommendation = f"No irrigation needed for this period due to sufficient rainfall."
        else:
            recommendation = (
                f"Apply approximately {total_irrigation_display:.1f} {display_unit} of irrigation "
                f"to replace water used by crops during this {period_days}-day period."
            )
            
            # Add frequency suggestion
            if period_days >= 7:
                weekly_avg = total_irrigation / (period_days / 7)
                weekly_avg_display = weekly_avg * conversion_factor
                recommendation += (
                    f" Consider applying approximately {weekly_avg_display:.1f} {display_unit} per week "
                    f"based on the average water use."
                )
        
        return {
            "status": "success",
            "mode": "daily",
            "period_days": period_days,
            "total_et": total_et_display,
            "total_rain": total_rain_display,
            "total_irrigation": total_irrigation_display,
            "units": display_unit,
            "irrigation_data": result_df["Irrigation_mm"].tolist(),
            "recommendation": recommendation
        }
    
    else:  # threshold mode
        # Threshold-based scheduling
        schedule = compute_threshold_schedule(
            df, 
            threshold_mm=threshold_mm, 
            et_col=et_col, 
            rain_col=rain_col, 
            efficiency=system_efficiency
        )
        
        # Convert schedule to display units
        conversion_factor = 1.0 if units == "mm" else 1/25.4  # mm to inches
        schedule_display = [
            (date, amount * conversion_factor) for date, amount in schedule
        ]
        
        total_irrigation = sum(amount for _, amount in schedule)
        total_irrigation_display = total_irrigation * conversion_factor
        display_unit = "mm" if units == "mm" else "inches"
        
        # Build recommendation text
        if len(schedule) == 0:
            recommendation = "No irrigation needed for this period based on the specified threshold."
        else:
            irrigation_events = len(schedule)
            recommendation = (
                f"Based on a depletion threshold of {threshold_mm * conversion_factor:.1f} {display_unit}, "
                f"irrigate {irrigation_events} time(s) during this {period_days}-day period, "
                f"for a total of {total_irrigation_display:.1f} {display_unit}."
            )
        
        return {
            "status": "success",
            "mode": "threshold",
            "threshold": threshold_mm * conversion_factor,
            "period_days": period_days,
            "irrigation_events": len(schedule),
            "total_et": total_et * conversion_factor,
            "total_rain": total_rain * conversion_factor,
            "total_irrigation": total_irrigation_display,
            "units": display_unit,
            "schedule": [
                {"date": date.strftime('%Y-%m-%d'), "amount": round(amount, 1)} 
                for date, amount in schedule_display
            ],
            "recommendation": recommendation
        }

def summarize_irrigation_needs(recommendation: Dict) -> str:
    """
    Create a human-readable summary of irrigation recommendations.
    
    Args:
        recommendation: Dictionary with recommendation information
        
    Returns:
        str: Formatted summary text
    """
    if recommendation["status"] != "success":
        return "Unable to generate irrigation recommendations. Please check your data."
    
    mode = recommendation["mode"]
    units = recommendation["units"]
    total_et = recommendation["total_et"]
    total_rain = recommendation["total_rain"]
    total_irrigation = recommendation["total_irrigation"]
    period_days = recommendation["period_days"]
    
    # Create summary text
    summary = []
    summary.append(f"Period analyzed: {period_days} days")
    summary.append(f"Total crop water use (ET): {total_et:.1f} {units}")
    
    if total_rain > 0:
        summary.append(f"Total rainfall: {total_rain:.1f} {units}")
        summary.append(f"Net irrigation need: {total_irrigation:.1f} {units}")
    else:
        summary.append(f"Total irrigation need: {total_irrigation:.1f} {units}")
    
    # Add recommendation
    summary.append("\nRecommendation:")
    summary.append(recommendation["recommendation"])
    
    # Add schedule details for threshold mode
    if mode == "threshold" and "schedule" in recommendation and recommendation["schedule"]:
        summary.append("\nIrrigation schedule:")
        for event in recommendation["schedule"]:
            summary.append(f"  • {event['date']}: Apply {event['amount']} {units}")
    
    return "\n".join(summary)