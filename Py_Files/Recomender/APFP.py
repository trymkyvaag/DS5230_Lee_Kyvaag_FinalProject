import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def prepare_data(df):
    """
    Prepare the dataframe for association rule mining.
    """
    df_processed = df.copy()

    df_processed['high_income'] = (
        df_processed['Income'] == '>50K').astype(str)
    categorical_features = [
        'Workclass', 'Education', 'Marital-status', 'Occupation',
        'Relationship', 'Race', 'Sex', 'Native-country'
    ]
    df_processed['Age_cat'] = pd.cut(df_processed['Age'], bins=[0, 25, 35, 45, 55, 100],
                                     labels=['Young', 'Early-Career', 'Mid-Career', 'Late-Career', 'Senior'])
    df_processed['Education-num_cat'] = pd.cut(df_processed['Education-num'],
                                               bins=[0, 8, 10, 12, 16, 100],
                                               labels=['Basic', 'Some-HS', 'HS-Grad', 'Bachelors', 'Advanced'])
    df_processed['Hours-per-week_cat'] = pd.cut(df_processed['Hours-per-week'],
                                                bins=[0, 20, 40, 60, 100],
                                                labels=['Part-time', 'Standard', 'Overtime', 'Extensive'])
    rule_features = categorical_features + \
        ['Age_cat', 'Education-num_cat', 'Hours-per-week_cat', 'high_income']
    return df_processed, rule_features


def encode_transactions(df, rule_features):
    """
    Convert the data into a one-hot encoded format suitable for frequent itemset mining.
    """
    te = TransactionEncoder()
    transactions = df[rule_features].apply(
        lambda x: [str(val) for val in x if pd.notna(val)], axis=1).tolist()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def mine_association_rules(te_df, algorithm='apriori', min_support=0.005, min_confidence=0.2, min_lift=1.0, max_lift=None):
    """
    Mine association rules using either Apriori or FP-Growth algorithm.
    """
    if algorithm == 'apriori':
        frequent_itemsets = apriori(
            te_df, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(
            te_df, min_support=min_support, use_colnames=True)
    rules = generate_association_rules(
        frequent_itemsets, min_confidence=min_confidence, min_lift=min_lift)
    

    return rules.sort_values(['lift', 'confidence'], ascending=False)


def print_rules(rules, top_n=10):
    """
    Print the top N rules in a readable format.
    """
    if rules.empty:
        print("\nNo rules found.")
    else:
        print(f"\nTop {top_n} Rules:")
        for idx, row in rules.head(top_n).iterrows():
            print(f"Rule #{idx+1}")
            print(f"  Antecedents: {', '.join(map(str, row['antecedents']))}")
            print(f"  Consequents: {', '.join(map(str, row['consequents']))}")
            print(
                f"  Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")


def generate_association_rules(frequent_itemsets, min_confidence=0.3, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.
    """
    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=-10)
    # print(rules.head())
    # Filter by lift
    high_lift_rules = rules[rules['lift'] <= 0.5] # CHange here as well

    return rules.sort_values(by=['lift', 'confidence'], ascending=True)


def create_feature_mapping(df, feature_columns):
    feature_map = {}
    for col in feature_columns:
        unique_values = df[col].dropna().unique()
        for value in unique_values:
            if isinstance(value, bool):
                feature_map[f'{value}'] = f"{col}: {'True' if value else 'False'}"
            else:
                feature_map[str(value)] = f"{col}: {value}"

    return feature_map




def filter_rules_by_metrics(rules, df, feature_columns, min_confidence=0, min_lift=-10):
    """
    Filter association rules based on confidence and lift, and map antecedents and consequents 
    to the feature columns for better clarity.
    """
    if len(rules) != 0:
        # For rules to be more readable, Col_name : value
        feature_map = create_feature_mapping(df, feature_columns)

        # CHange here if u want high vs low!
        filtered_rules = rules.sort_values(
            by=['lift', 'confidence'], ascending=[False, False])

        filtered_rules['antecedents_features'] = filtered_rules['antecedents'].apply(
            lambda x: [feature_map.get(str(item), str(item)) for item in x]) 

        filtered_rules['consequents_features'] = filtered_rules['consequents'].apply(
            lambda x: [feature_map.get(str(item), str(item)) for item in x]) 
        
        # income_related_terms = ['high_income: True', 'high_income: False', '>50K', '<=50K']
        high_income_rules = filtered_rules[
        filtered_rules["consequents_features"].apply(lambda x: any('high_income' in feature for feature in x))
        ]
        for idx, row in high_income_rules.iterrows():
            print(f"Rule #{idx + 1}: {row['antecedents']} -> {row['consequents']}")
        return filtered_rules
    else:
        return None

def write_rules_to_excel(rules_df, filename="filtered_rules.xlsx"):
    """
    Write the filtered rules DataFrame to an Excel file.
    """
    with pd.ExcelWriter(filename) as writer:
        rules_df.to_excel(writer, index=False)




def main():
    column_names = [
        "Age", "Workclass", "Fnlwgt", "Education", "Education-num",
        "Marital-status", "Occupation", "Relationship", "Race", "Sex",
        "Capital-gain", "Capital-loss", "Hours-per-week", "Native-country", "Income"
    ]
    df = pd.read_csv('Data/adult.csv', names=column_names,
                     skipinitialspace=True)
    processed_data, rule_features = prepare_data(df)
    te_df = encode_transactions(processed_data, rule_features)
    all_filtered_rules = []
    algorithms = ['fpgrowth']  # Test both Apriori and FP-Growth, FP was best
    support_values = [0.03]#, 0.2, 0.01]  
    confidence_values = [0.7]#, 0.5, 0.7]  

    for algorithm in algorithms:
        print(f"\n{'='*10} {algorithm.upper()} Analysis {'='*10}")
        for support in support_values:
            for confidence in confidence_values:
                print(f"\nSupport: {support}, Confidence: {confidence}")
                rules = mine_association_rules(
                    te_df, algorithm, min_support=support, min_confidence=confidence) 
                filtered_rules = filter_rules_by_metrics(
                    rules, processed_data, list(processed_data.columns))

                if filtered_rules is not None:
                    print(filtered_rules.head())
                    all_filtered_rules.append(filtered_rules)

    if all_filtered_rules:
        all_filtered_rules_df = pd.concat(
            all_filtered_rules, ignore_index=True)
        write_rules_to_excel(all_filtered_rules_df)


if __name__ == "__main__":
    main()
