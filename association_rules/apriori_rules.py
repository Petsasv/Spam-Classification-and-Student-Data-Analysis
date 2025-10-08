import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_for_apriori(df):
    """
    Preprocess the student performance data for Apriori algorithm.
    Converts numerical features to categorical bins and prepares data for association rules.
    
    Args:
        df: Original student performance dataframe
    
    Returns:
        Processed dataframe ready for Apriori
    """
    processed_df = df.copy()
    
    score_bins = [0, 60, 70, 80, 90, 100]
    score_labels = ['F', 'D', 'C', 'B', 'A']
    
    for score_col in ['math score', 'reading score', 'writing score']:
        grade_col = f'{score_col}_grade'
        processed_df[grade_col] = pd.cut(
            processed_df[score_col],
            bins=score_bins,
            labels=score_labels,
            include_lowest=True
        )
        for grade in score_labels:
            processed_df[f'{score_col}_{grade}'] = (processed_df[grade_col] == grade)
    
    categorical_cols = [
        'gender', 'race/ethnicity', 'parental level of education',
        'lunch', 'test preparation course'
    ]
    
    for col in categorical_cols:
        unique_values = processed_df[col].unique()
        for value in unique_values:
            processed_df[f'{col}_{value}'] = (processed_df[col] == value)
    
    columns_to_drop = (['math score', 'reading score', 'writing score'] + 
                      [f'{col}_grade' for col in ['math score', 'reading score', 'writing score']] +
                      categorical_cols)
    processed_df = processed_df.drop(columns=columns_to_drop)
    
    processed_df = processed_df.astype(bool)
    
    print("\nCreated binary features:")
    for col in processed_df.columns:
        support = processed_df[col].mean()
        print(f"{col}: {support:.3f}")
    
    return processed_df

def find_association_rules(df, min_support=0.1, min_lift=1.5):
    """
    Find association rules using Apriori algorithm
    
    Args:
        df: Preprocessed dataframe
        min_support: Minimum support threshold
        min_lift: Minimum lift threshold
    
    Returns:
        DataFrame with association rules
    """
    frequent_itemsets = apriori(
        df,
        min_support=min_support,
        use_colnames=True
    )
    
    print("\nFrequent Itemsets Analysis:")
    print("="*50)
    print(f"Total number of frequent itemsets: {len(frequent_itemsets)}")
    print("\nSupport distribution:")
    print(frequent_itemsets['support'].describe())
    
    print("\nExample itemsets (first 5):")
    for idx, row in frequent_itemsets.head().iterrows():
        items = list(row['itemsets'])
        print(f"\nItemset {idx + 1}:")
        print(f"Items: {items}")
        print(f"Support: {row['support']:.3f}")
    
    try:
        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=min_lift
        )
        
        print("\nAssociation Rules Analysis:")
        print("="*50)
        print(f"Total number of rules: {len(rules)}")
        if not rules.empty:
            print("\nRule metrics distribution:")
            print(rules[['support', 'confidence', 'lift']].describe())
            
            print("\nTop 3 rules by lift:")
            for idx, row in rules.nlargest(3, 'lift').iterrows():
                print(f"\nRule {idx + 1}:")
                print(f"IF {list(row['antecedents'])}")
                print(f"THEN {list(row['consequents'])}")
                print(f"Support: {row['support']:.3f}")
                print(f"Confidence: {row['confidence']:.3f}")
                print(f"Lift: {row['lift']:.3f}")
        
    except Exception as e:
        print(f"\nWarning: Error generating rules: {str(e)}")
        print("This might be due to insufficient support or lift in the itemsets.")
        return None
    
    if not rules.empty:
        rules = rules.sort_values('lift', ascending=False)
    
    return rules

def analyze_rules(rules, top_n=10):
    """
    Analyze and print interesting association rules
    
    Args:
        rules: DataFrame with association rules
        top_n: Number of top rules to analyze
    """
    os.makedirs('results', exist_ok=True)
    
    rules.to_csv('results/all_rules.csv', index=False)
    top_rules = rules.nlargest(top_n, 'lift')
    
    with open('results/top_rules.txt', 'w', encoding='utf-8') as f:
        f.write("=== Top {} Association Rules by Lift ===\n\n".format(top_n))
        f.write("Total rules found: {}\n".format(len(rules)))
        f.write("Support range: {:.3f} to {:.3f}\n".format(rules['support'].min(), rules['support'].max()))
        f.write("Confidence range: {:.3f} to {:.3f}\n".format(rules['confidence'].min(), rules['confidence'].max()))
        f.write("Lift range: {:.3f} to {:.3f}\n\n".format(rules['lift'].min(), rules['lift'].max()))
        
        for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
            antecedents = ', '.join(rule['antecedents'])
            consequents = ', '.join(rule['consequents'])
            f.write(f"Rule {i}:\n")
            f.write(f"IF {antecedents}\n")
            f.write(f"THEN {consequents}\n")
            f.write(f"Support: {rule['support']:.3f}\n")
            f.write(f"Confidence: {rule['confidence']:.3f}\n")
            f.write(f"Lift: {rule['lift']:.3f}\n\n")
    
    return top_rules

def visualize_rules(rules, top_n=20):
    """
    Create visualizations for the association rules
    
    Args:
        rules: DataFrame with association rules
        top_n: Number of top rules to visualize
    """
    os.makedirs('results', exist_ok=True)
    top_rules = rules.nlargest(top_n, 'lift')
    
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(rules['support'], rules['confidence'], 
                         c=rules['lift'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (colored by Lift)')
    
    plt.subplot(2, 2, 2)
    plt.barh(range(top_n), top_rules['lift'])
    plt.yticks(range(top_n), [f"Rule {i+1}" for i in range(top_n)])
    plt.xlabel('Lift')
    plt.title(f'Top {top_n} Rules by Lift')
    
    plt.subplot(2, 2, 3)
    plt.hist(rules['support'], bins=30, alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Frequency')
    plt.title('Distribution of Support')
    
    plt.subplot(2, 2, 4)
    plt.hist(rules['confidence'], bins=30, alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence')
    
    plt.tight_layout()
    plt.savefig('results/rule_visualizations.png')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 8))
    metrics = ['support', 'confidence', 'lift']
    plt.imshow(top_rules[metrics].values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.xticks(range(len(metrics)), metrics, rotation=45)
    plt.yticks(range(top_n), [f"Rule {i+1}" for i in range(top_n)])
    plt.title('Heatmap of Rule Metrics')
    plt.tight_layout()
    plt.savefig('results/rule_metrics_heatmap.png')
    plt.close(fig)

def main():
    """Main function to run the association rules analysis"""
    print("Loading student performance dataset...")
    df = pd.read_csv('data/StudentsPerformance.csv')
    
    print("\nPreprocessing data for association rules...")
    processed_df = preprocess_for_apriori(df)
    
    print("\nPreprocessed data shape:", processed_df.shape)
    print("Number of binary features:", processed_df.shape[1])
    
    print("\nFinding association rules...")
    rules = find_association_rules(processed_df, min_support=0.1, min_lift=1.5)
    
    return rules

if __name__ == "__main__":
    main() 