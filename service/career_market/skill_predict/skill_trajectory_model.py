import pandas as pd
import json
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
import joblib

# Suppress logs
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def parse_skill_counts(row):
    try:
        if pd.isna(row) or not row:
            return {}
        return json.loads(row.replace('""', '"'))
    except:
        return {}

import os

def main():
    # Resolve the path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'may3.csv')
    
    print(f"Loading dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f" Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)

    print("Processing timestamps and skill data...")
    df['ds'] = pd.to_datetime(df['ran_at']).dt.tz_localize(None).dt.floor('D')
    df['skills'] = df['skill_counts'].apply(parse_skill_counts)

    # Convert to long format
    skill_data = []
    for _, row in df.iterrows():
        for skill, count in row['skills'].items():
            skill_data.append({
                'ds': row['ds'],
                'skill': skill,
                'count': count
            })

    skill_df = pd.DataFrame(skill_data)

    #  STEP 1: DAILY AVERAGE (handles multiple rows per day)
    skill_series = (
        skill_df.groupby(['ds', 'skill'])['count']
        .mean()   # ✅ AVERAGE instead of sum
        .reset_index()
    )

    #  STEP 2: FILL MISSING DATES
    print("Fixing time series gaps...")
    full_dates = pd.date_range(skill_series['ds'].min(), skill_series['ds'].max())

    fixed_data = []
    for skill in skill_series['skill'].unique():
        temp = skill_series[skill_series['skill'] == skill].set_index('ds')['count']
        temp = temp.reindex(full_dates, fill_value=0)
        temp = temp.rename_axis('ds').reset_index()
        temp['skill'] = skill
        fixed_data.append(temp)

    skill_series = pd.concat(fixed_data)

    #  STEP 3: REMOVE WEAK SKILLS
    skill_totals = skill_series.groupby('skill')['count'].sum()
    target_skills = skill_totals[skill_totals >= 300].index.tolist()

    print(f"Total skills to forecast: {len(target_skills)}")

    print("\n" + "="*90)
    print(f"{'Skill':<20} | {'RMSE':<8} | {'MAPE %':<8} | {'Coverage %':<12} | {'Growth %':<10} | Status")
    print("-"*90)

    results = []
    trained_models = {}

    for skill in target_skills:
        df_skill_raw = skill_series[skill_series['skill'] == skill][['ds', 'count']]
        latest = df_skill_raw['count'].iloc[-1]

        df_skill = df_skill_raw.rename(columns={'count': 'y'})

        #  STEP 4: SMOOTHING
        df_skill['y'] = df_skill['y'].rolling(window=3, min_periods=1).mean()

        #  STEP 5: LOG TRANSFORM
        df_skill['y'] = np.log1p(df_skill['y'])

        #  STEP 6: IMPROVED PROPHET
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.03,
            seasonality_prior_scale=1.5
        )

        model.fit(df_skill)
        trained_models[skill] = model

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        #  STEP 7: SAFE INVERSE TRANSFORM
        forecast['yhat'] = np.expm1(forecast['yhat'])

        # forecast['yhat'] already computed above


        #  CLIP EXTREME VALUES
        forecast['yhat'] = np.clip(forecast['yhat'], 0, latest * 5)

        forecast_value = forecast['yhat'].iloc[-1]

        #  STEP 8: EVALUATION
        rmse, mape, coverage = None, None, None
        try:
            df_cv = cross_validation(
                model,
                initial='120 days',
                period='30 days',
                horizon='30 days'
            )
            df_p = performance_metrics(df_cv)

            rmse = round(df_p['rmse'].mean(), 2)
            mape = round(df_p['mape'].mean() * 100, 2)
            coverage = round(df_p['coverage'].mean() * 100, 2)
        except:
            pass

        #  STEP 9: GROWTH & TREND
        delta = forecast_value - latest
        
        # absolute growth
        absolute_growth = delta
        # relative growth (%)
        relative_growth_pct = ((delta / latest) * 100) if latest > 0 else 0
        #  FINAL CONSISTENT SCORE (Log-scaled growth)
        growth_score = np.log1p(forecast_value) - np.log1p(latest)

        # Adaptive threshold (research-grade)
        threshold = max(1, 0.1 * latest)

        if delta > threshold:
            status = "Rising"
        elif delta < -threshold:
            status = "Declining"
        else:
            status = "Stable"

        print(f"{skill:<20} | {str(rmse):<8} | {str(mape):<8} | {str(coverage):<12} | {round(growth_score,4):<12} | {status}")

        results.append({
            'skill_name': skill,
            'current_count': round(latest, 2),
            'forecasted_count_30d': round(forecast_value, 2),
            'absolute_growth': round(absolute_growth, 2),
            'relative_growth_pct': round(relative_growth_pct, 2),
            'growth_score': round(growth_score, 4),
            'trend_status': status,
            'rmse': rmse,
            'mape_pct': mape,
            'coverage_pct': coverage
        })

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(script_dir, 'fixed_skill_forecast.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    #  NEW: Save top 5 rising and declining for frontend
    top_rising = results_df[results_df['trend_status'] == 'Rising'].sort_values('growth_score', ascending=False).head(5)
    top_declining = results_df[results_df['trend_status'] == 'Declining'].sort_values('growth_score', ascending=True).head(5)
    
    trends_json = {
        "rising": top_rising[['skill_name', 'growth_score', 'absolute_growth', 'trend_status', 'coverage_pct', 'mape_pct']].to_dict(orient='records'),
        "declining": top_declining[['skill_name', 'growth_score', 'absolute_growth', 'trend_status', 'coverage_pct', 'mape_pct']].to_dict(orient='records')
    }
    
    trends_json_path = os.path.join(script_dir, 'top_trends.json')
    with open(trends_json_path, 'w') as f:
        json.dump(trends_json, f, indent=4)
    print(f"Top trends saved to {trends_json_path}")

    #  STEP 10: SAVE MODELS
    print("\nSaving models...")
    models_path = os.path.join(script_dir, 'skill_models.joblib')
    joblib.dump(trained_models, models_path)
    print(f"All models saved to {models_path}")
    
    print("\n" + "="*90)
    print("FINAL SUMMARY REPORT")
    print("="*90)
    
    # Top Rising
    rising = results_df[results_df['trend_status'] == 'Rising'].sort_values('growth_score', ascending=False).head(5)
    print("\n TOP 5 RISING SKILLS:")
    for _, row in rising.iterrows():
        print(f"  - {row['skill_name']:<20}: Score {row['growth_score']:.4f} (+{row['absolute_growth']:.1f} units)")
        
    # Top Declining
    declining = results_df[results_df['trend_status'] == 'Declining'].sort_values('growth_score', ascending=True).head(5)
    print("\n TOP 5 DECLINING SKILLS:")
    for _, row in declining.iterrows():
        print(f"  - {row['skill_name']:<20}: Score {row['growth_score']:.4f} ({row['absolute_growth']:.1f} units)")

    # Model Performance
    avg_rmse = results_df['rmse'].dropna().mean()
    avg_mape = results_df['mape_pct'].dropna().mean()
    avg_cov = results_df['coverage_pct'].dropna().mean()
    
    print("\n GLOBAL MODEL PERFORMANCE:")
    print(f"  - Avg RMSE:     {avg_rmse:.2f}")
    print(f"  - Avg MAPE:     {avg_mape:.2f}%")
    print(f"  - Avg Coverage: {avg_cov:.2f}%")
    
    print("\n Done. Results saved to fixed_skill_forecast.csv")

if __name__ == "__main__":
    main()