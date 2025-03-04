"""
Unit tests for the irrigation module.
"""
import os
import sys
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irrigation import (
    compute_daily_replacement,
    compute_threshold_schedule,
    get_irrigation_recommendation,
    summarize_irrigation_needs
)

class TestIrrigation(unittest.TestCase):
    """Test cases for the irrigation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample dataframe with 10 days of data
        dates = [datetime(2023, 7, 1) + timedelta(days=i) for i in range(10)]
        et_values = [4.0, 4.5, 5.0, 5.2, 5.5, 5.3, 4.8, 4.6, 4.2, 4.0]  # in mm
        rain_values = [0.0, 0.0, 2.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # in mm
        
        self.df = pd.DataFrame({
            'date': dates,
            'ET': et_values,
            'Rain': rain_values
        })
    
    def test_compute_daily_replacement_no_rain(self):
        """Test daily replacement calculation with no rain data."""
        # Create a dataframe without rain
        df_no_rain = self.df[['date', 'ET']].copy()
        
        # Calculate daily replacement
        result = compute_daily_replacement(df_no_rain, et_col='ET')
        
        # Check that irrigation equals ET
        self.assertTrue('Irrigation_mm' in result.columns)
        self.assertEqual(result['Irrigation_mm'].tolist(), result['ET'].tolist())
        
        # Check efficiency applies correctly
        result_with_efficiency = compute_daily_replacement(df_no_rain, et_col='ET', efficiency=0.8)
        
        # Irrigation should be ET / efficiency
        for i, row in result_with_efficiency.iterrows():
            self.assertAlmostEqual(row['Irrigation_mm'], row['ET'] / 0.8)
    
    def test_compute_daily_replacement_with_rain(self):
        """Test daily replacement calculation with rain data."""
        # Calculate daily replacement
        result = compute_daily_replacement(self.df, et_col='ET', rain_col='Rain')
        
        # Check results for a few days
        # Day 1: ET=4.0, Rain=0.0 -> Net_ET=4.0
        self.assertAlmostEqual(result.iloc[0]['Net_ET'], 4.0)
        self.assertAlmostEqual(result.iloc[0]['Irrigation_mm'], 4.0)
        
        # Day 3: ET=5.0, Rain=2.0 -> Net_ET=3.0
        self.assertAlmostEqual(result.iloc[2]['Net_ET'], 3.0)
        self.assertAlmostEqual(result.iloc[2]['Irrigation_mm'], 3.0)
        
        # Day 4: ET=5.2, Rain=8.0 -> Net_ET=0.0 (rain exceeds ET)
        self.assertAlmostEqual(result.iloc[3]['Net_ET'], 0.0)
        self.assertAlmostEqual(result.iloc[3]['Irrigation_mm'], 0.0)
        
        # Check with efficiency
        result_with_efficiency = compute_daily_replacement(
            self.df, et_col='ET', rain_col='Rain', efficiency=0.75
        )
        
        # Day 1: ET=4.0, Rain=0.0 -> Net_ET=4.0 -> Irrigation=4.0/0.75=5.33
        self.assertAlmostEqual(result_with_efficiency.iloc[0]['Irrigation_mm'], 4.0/0.75)
    
    def test_compute_threshold_schedule(self):
        """Test threshold-based irrigation scheduling."""
        # Set threshold to 10 mm
        threshold = 10.0
        
        # Compute schedule
        schedule = compute_threshold_schedule(
            self.df, threshold_mm=threshold, et_col='ET', rain_col='Rain'
        )
        
        # Check schedule
        # With our data, the deficit accumulates:
        # Day 1: 4.0 -> deficit=4.0
        # Day 2: 4.5 -> deficit=8.5
        # Day 3: 5.0-2.0=3.0 -> deficit=11.5 > 10 -> irrigate 11.5
        # Day 4: 5.2-8.0=-2.8 -> deficit=0 (rain exceeds ET)
        # etc.
        
        # First irrigation should be on day 3
        self.assertEqual(schedule[0][0], self.df['date'].iloc[2])
        self.assertAlmostEqual(schedule[0][1], 11.5)
        
        # Try with a larger threshold
        threshold = 15.0
        schedule = compute_threshold_schedule(
            self.df, threshold_mm=threshold, et_col='ET', rain_col='Rain'
        )
        
        # Should have fewer irrigation events
        self.assertLess(len(schedule), 5)  # Just a sanity check
        
        # Try with efficiency
        schedule = compute_threshold_schedule(
            self.df, threshold_mm=10.0, et_col='ET', rain_col='Rain', efficiency=0.8
        )
        
        # First event irrigation amount should be higher due to efficiency
        self.assertAlmostEqual(schedule[0][1], 11.5 / 0.8)
    
    def test_get_irrigation_recommendation_daily(self):
        """Test getting daily irrigation recommendations."""
        rec = get_irrigation_recommendation(
            self.df,
            mode="daily",
            system_efficiency=0.9,
            et_col="ET",
            rain_col="Rain",
            units="mm"
        )
        
        # Check recommendation structure
        self.assertEqual(rec["status"], "success")
        self.assertEqual(rec["mode"], "daily")
        self.assertEqual(rec["units"], "mm")
        
        # Calculate expected total irrigation (sum of ET-Rain, with efficiency)
        expected_total = sum(max(0, et - rain) for et, rain in zip(self.df['ET'], self.df['Rain'])) / 0.9
        self.assertAlmostEqual(rec["total_irrigation"], expected_total)
        
        # Test with different units
        rec_inches = get_irrigation_recommendation(
            self.df,
            mode="daily",
            system_efficiency=0.9,
            et_col="ET",
            rain_col="Rain",
            units="inches"
        )
        
        # Should be same value but in inches
        self.assertEqual(rec_inches["units"], "inches")
        self.assertAlmostEqual(rec_inches["total_irrigation"], expected_total / 25.4)
    
    def test_get_irrigation_recommendation_threshold(self):
        """Test getting threshold-based irrigation recommendations."""
        threshold = 12.0
        rec = get_irrigation_recommendation(
            self.df,
            mode="threshold",
            threshold_mm=threshold,
            system_efficiency=1.0,  # No efficiency loss
            et_col="ET",
            rain_col="Rain",
            units="mm"
        )
        
        # Check recommendation structure
        self.assertEqual(rec["status"], "success")
        self.assertEqual(rec["mode"], "threshold")
        self.assertEqual(rec["threshold"], threshold)
        
        # Should have a schedule
        self.assertIn("schedule", rec)
        
        # Calculate manually how many times threshold would be exceeded
        deficit = 0
        irrigation_events = 0
        total_irrigation = 0
        
        for et, rain in zip(self.df['ET'], self.df['Rain']):
            deficit += max(0, et - rain)
            if deficit >= threshold:
                irrigation_events += 1
                total_irrigation += deficit
                deficit = 0
        
        # If there's remaining deficit at the end, include it
        if deficit > 0:
            irrigation_events += 1
            total_irrigation += deficit
        
        self.assertEqual(rec["irrigation_events"], irrigation_events)
        self.assertAlmostEqual(rec["total_irrigation"], total_irrigation)
    
    def test_summarize_irrigation_needs(self):
        """Test creating irrigation summary text."""
        # Create a simple recommendation
        rec = {
            "status": "success",
            "mode": "daily",
            "period_days": 10,
            "total_et": 47.1,
            "total_rain": 11.0,
            "total_irrigation": 40.1,
            "units": "mm",
            "recommendation": "Apply approximately 40.1 mm of irrigation to replace water used by crops."
        }
        
        # Get summary
        summary = summarize_irrigation_needs(rec)
        
        # Check that it contains key information
        self.assertIn("Period analyzed: 10 days", summary)
        self.assertIn("Total crop water use (ET): 47.1 mm", summary)
        self.assertIn("Total rainfall: 11.0 mm", summary)
        self.assertIn("Net irrigation need: 40.1 mm", summary)
        
        # Test with threshold mode and schedule
        rec = {
            "status": "success",
            "mode": "threshold",
            "period_days": 10,
            "total_et": 47.1,
            "total_rain": 11.0,
            "total_irrigation": 40.1,
            "units": "mm",
            "recommendation": "Based on a depletion threshold...",
            "schedule": [
                {"date": "2023-07-03", "amount": 12.5},
                {"date": "2023-07-08", "amount": 27.6}
            ]
        }
        
        summary = summarize_irrigation_needs(rec)
        
        # Should include schedule
        self.assertIn("Irrigation schedule:", summary)
        self.assertIn("2023-07-03: Apply 12.5 mm", summary)
        self.assertIn("2023-07-08: Apply 27.6 mm", summary)

if __name__ == '__main__':
    unittest.main()