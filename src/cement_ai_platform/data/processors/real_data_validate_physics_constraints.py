import pandas as pd
import numpy as np

print("🔬 VALIDATING DATA AGAINST PHYSICS CONSTRAINTS")
print("=" * 50)

validation_results = {
    'total_records': len(integrated_df),
    'violations': {},
    'warnings': {},
    'passed_checks': [],
    'failed_checks': []
}

print(f"Validating {len(integrated_df)} integrated records...")

# 1. Composition constraints (Kaggle data)
kaggle_data = integrated_df[integrated_df['data_source'] == 'Kaggle'].copy()
print(f"\n1️⃣ COMPOSITION CONSTRAINTS ({len(kaggle_data)} samples)")

if len(kaggle_data) > 0:
    # Water-cement ratio constraints
    w_c_violations = kaggle_data[(kaggle_data['w_c_ratio'] < 0.2) | (kaggle_data['w_c_ratio'] > 1.0)]
    if len(w_c_violations) > 0:
        validation_results['violations']['w_c_ratio'] = len(w_c_violations)
        validation_results['failed_checks'].append('Water-cement ratio (0.2-1.0)')
        print(f"  ❌ Water-cement ratio: {len(w_c_violations)} violations (range: 0.2-1.0)")
    else:
        validation_results['passed_checks'].append('Water-cement ratio (0.2-1.0)')
        print(f"  ✅ Water-cement ratio: All values within acceptable range")

    # Compressive strength constraints
    strength_violations = kaggle_data[(kaggle_data['compressive_strength_mpa'] < 0) | (kaggle_data['compressive_strength_mpa'] > 200)]
    if len(strength_violations) > 0:
        validation_results['violations']['compressive_strength'] = len(strength_violations)
        validation_results['failed_checks'].append('Compressive strength (0-200 MPa)')
        print(f"  ❌ Compressive strength: {len(strength_violations)} violations (range: 0-200 MPa)")
    else:
        validation_results['passed_checks'].append('Compressive strength (0-200 MPa)')
        print(f"  ✅ Compressive strength: All values within acceptable range")

    # Slump test constraints
    slump_violations = kaggle_data[(kaggle_data['slump_cm'] < 0) | (kaggle_data['slump_cm'] > 30)]
    _slump_violations_count = slump_violations.dropna().shape[0] if 'slump_cm' in slump_violations.columns else 0
    if _slump_violations_count > 0:
        validation_results['violations']['slump'] = _slump_violations_count
        validation_results['failed_checks'].append('Slump test (0-30 cm)')
        print(f"  ❌ Slump test: {_slump_violations_count} violations (range: 0-30 cm)")
    else:
        validation_results['passed_checks'].append('Slump test (0-30 cm)')
        print(f"  ✅ Slump test: All values within acceptable range")

# 2. Raw material composition constraints (Global data)
global_data = integrated_df[integrated_df['data_source'] == 'Global_DB'].copy()
print(f"\n2️⃣ RAW MATERIAL CONSTRAINTS ({len(global_data)} facilities)")

if len(global_data) > 0:
    # Check if raw material percentages sum approximately to 100%
    raw_material_cols = ['limestone_pct', 'clay_pct', 'iron_ore_pct', 'silica_sand_pct', 'gypsum_pct']
    _valid_rows = global_data[raw_material_cols].dropna()
    if len(_valid_rows) > 0:
        _raw_material_sums = _valid_rows.sum(axis=1)
        _sum_violations = _valid_rows[(_raw_material_sums < 90) | (_raw_material_sums > 110)]

        if len(_sum_violations) > 0:
            validation_results['warnings']['raw_material_sum'] = len(_sum_violations)
            print(f"  ⚠️  Raw material sum: {len(_sum_violations)} facilities outside 90-110% range")
        else:
            validation_results['passed_checks'].append('Raw material percentage sum (90-110%)')
            print(f"  ✅ Raw material percentages: All facilities within acceptable range")

# 3. Environmental constraints
print(f"\n3️⃣ ENVIRONMENTAL CONSTRAINTS")

if len(global_data) > 0:
    # CO2 emissions constraints (typical range: 500-1200 kg/t)
    co2_violations = global_data[(global_data['co2_emissions_kg_t'] < 400) | (global_data['co2_emissions_kg_t'] > 1500)]
    _co2_violations_count = co2_violations.dropna().shape[0]
    if _co2_violations_count > 0:
        validation_results['warnings']['co2_emissions'] = _co2_violations_count
        print(f"  ⚠️  CO2 emissions: {_co2_violations_count} facilities outside typical range (400-1500 kg/t)")
    else:
        validation_results['passed_checks'].append('CO2 emissions (400-1500 kg/t)')
        print(f"  ✅ CO2 emissions: All values within typical range")

    # Energy efficiency constraints (typical range: 2.5-5.0 GJ/t)
    energy_violations = global_data[(global_data['energy_efficiency_gj_t'] < 2.0) | (global_data['energy_efficiency_gj_t'] > 6.0)]
    _energy_violations_count = energy_violations.dropna().shape[0]
    if _energy_violations_count > 0:
        validation_results['warnings']['energy_efficiency'] = _energy_violations_count
        print(f"  ⚠️  Energy efficiency: {_energy_violations_count} facilities outside typical range (2.0-6.0 GJ/t)")
    else:
        validation_results['passed_checks'].append('Energy efficiency (2.0-6.0 GJ/t)')
        print(f"  ✅ Energy efficiency: All values within typical range")

# 4. Coordinate constraints
print(f"\n4️⃣ GEOGRAPHIC CONSTRAINTS")

coord_data = integrated_df[~integrated_df['latitude'].isna() & ~integrated_df['longitude'].isna()]
if len(coord_data) > 0:
    lat_violations = coord_data[(coord_data['latitude'] < -90) | (coord_data['latitude'] > 90)]
    lon_violations = coord_data[(coord_data['longitude'] < -180) | (coord_data['longitude'] > 180)]

    if len(lat_violations) > 0 or len(lon_violations) > 0:
        validation_results['violations']['coordinates'] = len(lat_violations) + len(lon_violations)
        validation_results['failed_checks'].append('Geographic coordinates')
        print(f"  ❌ Coordinates: {len(lat_violations)} latitude + {len(lon_violations)} longitude violations")
    else:
        validation_results['passed_checks'].append('Geographic coordinates')
        print(f"  ✅ Geographic coordinates: All values within valid ranges")

# 5. Summary
print(f"\n📊 VALIDATION SUMMARY")
print("=" * 30)
print(f"Total records validated: {validation_results['total_records']}")
print(f"Checks passed: {len(validation_results['passed_checks'])}")
print(f"Checks failed: {len(validation_results['failed_checks'])}")
print(f"Warnings issued: {len(validation_results['warnings'])}")

if validation_results['failed_checks']:
    print(f"\n❌ Failed checks:")
    for check in validation_results['failed_checks']:
        print(f"  • {check}")

if validation_results['warnings']:
    print(f"\n⚠️  Warnings:")
    for warning, count in validation_results['warnings'].items():
        print(f"  • {warning}: {count} records")

physics_validation_passed = len(validation_results['failed_checks']) == 0
print(f"\n🎯 Physics validation: {'✅ PASSED' if physics_validation_passed else '❌ FAILED'}")