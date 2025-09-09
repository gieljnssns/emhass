# EMHASS Deferrable Load Profiles

This document describes the new deferrable load profile functionality in EMHASS that allows you to "teach" the system about real usage patterns of your deferrable loads and use these learned profiles in optimization.

## Overview

Instead of using fixed nominal power values for deferrable loads, you can now:
1. **Learn** actual consumption profiles from historical Home Assistant data
2. **Store** these profiles for later use
3. **Apply** the learned profiles in optimization to get more realistic results

## New Functionality

### 1. Teaching Deferrable Load Profiles

The new `teach-deferrable` action allows you to capture the actual power consumption pattern of a deferrable load during a specific time window.

#### API Endpoint
```
POST http://localhost:5000/action/teach-deferrable
```

#### Runtime Parameters
- `start_time` (required): Start hour of the learning window (0-23)
- `end_time` (required): End hour of the learning window (0-23)
- `deferrable_load_id` (required): Home Assistant entity ID for the deferrable load
- `profile_name` (optional): Name for the saved profile (defaults to deferrable_load_id)
- `days_to_retrieve` (optional): Number of historical days to analyze (default: 7)
- `sensor_name` (optional): Sensor name if different from deferrable_load_id

#### Example Request
```json
{
  "start_time": 10,
  "end_time": 14,
  "deferrable_load_id": "sensor.washing_machine_power",
  "profile_name": "washing_machine_profile",
  "days_to_retrieve": 7,
  "sensor_name": "sensor.washing_machine_power"
}
```

### 2. Using Learned Profiles in Optimization

You can now use learned profiles in any optimization action by passing the `def_profile` parameter.

#### Runtime Parameter
- `def_profile`: List of profile names to use for each deferrable load (use `null` for loads that should use nominal power)

#### Example: Day-ahead Optimization with Profile
```json
{
  "def_profile": ["washing_machine_profile", null, "dishwasher_profile"]
}
```

This example:
- Uses `washing_machine_profile` for deferrable load 0
- Uses nominal power (traditional behavior) for deferrable load 1
- Uses `dishwasher_profile` for deferrable load 2

## File Storage

Learned profiles are stored in the `data/deferrable_profiles/` directory as pickle files with the following structure:

```python
{
    "profile_name": "washing_machine_profile",
    "sensor_name": "sensor.washing_machine_power",
    "start_time": 10,
    "end_time": 14,
    "days_retrieved": 7,
    "created_timestamp": "2024-01-15T10:30:00Z",
    "load_profile": [1200, 1180, 1250, ...],  # Power values in watts
    "statistics": {
        "mean_power": 1150.5,
        "max_power": 1400.2,
        "total_energy": 2.8,  # kWh
        "num_datapoints": 24
    },
    "time_step_minutes": 30
}
```

## Optimization Behavior

When using learned profiles:

1. **Power Bounds**: The maximum power from the learned profile replaces the nominal power for upper bounds
2. **Energy Constraints**: The average power from the learned profile is used for energy constraint calculations
3. **Sequence Optimization**: The learned profile is treated as a power sequence that the optimizer can schedule optimally

## Use Cases

### 1. Washing Machine
- Learn the actual wash cycle pattern (high power wash, lower power spin, etc.)
- Optimize start time based on real consumption pattern

### 2. Dishwasher
- Capture the multi-phase operation (pre-wash, wash, rinse, dry)
- Better energy planning with realistic power curves

### 3. Electric Vehicle Charging
- Learn charging curves for different charge levels
- Optimize charging schedule based on actual charging behavior

### 4. Heat Pump Water Heater
- Capture heating cycles and standby consumption
- Optimize heating schedule based on real thermal behavior

## Example Workflow

1. **Collect Data**: Let your deferrable load operate normally for a week
2. **Learn Profile**: Use the teach-deferrable action to capture the usage pattern
3. **Optimize**: Use the learned profile in day-ahead or MPC optimization
4. **Refine**: Update profiles periodically to account for seasonal changes

```bash
# Step 1: Teach the washing machine profile
curl -X POST http://localhost:5000/action/teach-deferrable \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": 8,
    "end_time": 18,
    "deferrable_load_id": "sensor.washing_machine_power",
    "profile_name": "washing_machine_weekday",
    "days_to_retrieve": 14
  }'

# Step 2: Use in optimization
curl -X POST http://localhost:5000/action/dayahead-optim \
  -H "Content-Type: application/json" \
  -d '{
    "def_profile": ["washing_machine_weekday"]
  }'
```

## Benefits

1. **Realistic Optimization**: Uses actual consumption patterns instead of fixed nominal values
2. **Better Energy Planning**: Accounts for varying power consumption during operation
3. **Improved Scheduling**: Optimizer understands the real duration and power requirements
4. **Adaptive**: Profiles can be updated to reflect changes in appliance behavior

## Compatibility

- Fully backward compatible - existing configurations continue to work
- Can mix learned profiles with nominal power values
- Works with all optimization methods (perfect, day-ahead, MPC)
- Integrates with existing deferrable load constraints (start times, end times, operating hours)

## Troubleshooting

### Profile Not Loading
- Check that the profile file exists in `data/deferrable_profiles/`
- Verify the profile name matches exactly (case-sensitive)
- Check EMHASS logs for detailed error messages

### Poor Optimization Results
- Ensure the learning time window captures complete operation cycles
- Consider using more historical days for better statistical representation
- Verify that the learned profile represents typical operation

### Data Quality Issues
- Check Home Assistant sensor data quality during the learning period
- Ensure the sensor accurately reflects the load's power consumption
- Consider filtering out outliers or unusual operation periods
