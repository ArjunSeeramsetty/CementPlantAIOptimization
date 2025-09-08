# Train models for all target variables
print("🚀 Training comprehensive cement quality prediction models")
print("Dataset shape:", cement_dataset.shape)
print("\nTarget variables to predict:")
print("1. Free lime content (% unreacted CaO)")
print("2. C3S content (primary strength compound)")  
print("3. Compressive strength (physics-derived)")

# Train all models
predictor.train_all_targets(cement_dataset)

# Display comprehensive results summary
print("\n" + "="*80)
print("📊 COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
print("="*80)

targets = ['free_lime', 'c3s_content', 'compressive_strength']
target_display = {
    'free_lime': 'Free Lime Content (%)',
    'c3s_content': 'C3S Content (%)',
    'compressive_strength': 'Compressive Strength (MPa)'
}

for target in targets:
    if target in predictor.model_performance:
        print(f"\n🎯 {target_display[target]}")
        print("-" * 50)
        
        results = predictor.model_performance[target]
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_r2 = results[best_model]['test_r2']
        
        # Display all model results
        print(f"{'Model':<18} {'Test R²':<10} {'Test RMSE':<12} {'CV Mean':<10} {'Status':<10}")
        print("-" * 70)
        
        for model_name, metrics in results.items():
            status = "🏆 BEST" if model_name == best_model else ""
            if metrics['test_r2'] > 0.8:
                status += " ✅ R²>0.8"
            
            print(f"{model_name:<18} {metrics['test_r2']:<10.3f} "
                  f"{metrics['test_rmse']:<12.3f} {metrics['cv_mean']:<10.3f} {status:<10}")
        
        print(f"\n🏆 Best model: {best_model} (R² = {best_r2:.3f})")
        
        # Top 5 features
        if target in predictor.feature_importance:
            top_features = predictor.feature_importance[target].head(5)
            print(f"\n📈 Top 5 important features for {target_display[target]}:")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print(f"📈 Successfully trained {len(predictor.models)} algorithms on {len(targets)} quality parameters")

# Check if R² > 0.8 target achieved
high_performance_count = 0
for target in targets:
    if target in predictor.model_performance:
        results = predictor.model_performance[target]
        best_r2 = max([r['test_r2'] for r in results.values()])
        if best_r2 > 0.8:
            high_performance_count += 1

print(f"🎯 Target Achievement: {high_performance_count}/{len(targets)} quality parameters achieved R² > 0.8")
if high_performance_count == len(targets):
    print("🏆 SUCCESS: All quality parameters achieved target accuracy!")
else:
    print("⚡ Partial success: Further optimization may be needed for remaining targets")

print("\n📊 Ready for detailed analysis and visualization...")