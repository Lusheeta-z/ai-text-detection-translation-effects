import pandas as pd
from pathlib import Path

# Read AI_Human CSV File
ai_human = pd.read_csv('AI_Human.csv')
# Read Balanced AI Human Prompts CSV file
balanced_ai_human_prompts = pd.read_csv('balanced_ai_human_prompts.csv')

print(ai_human.head())
print(balanced_ai_human_prompts.head())

# Word count of provided text
def word_count(text):
    if isinstance(text, str):
        return len(text.split())
    return 0

# Is text longer than min words required
def long_mask(df, min_words=70):
    return df['text'].apply(word_count) >= min_words

# Filter our AI generated essays from AI Human CSV
ai_generated_essays = ai_human[ai_human['generated'] == 1.0]
# Filter our Human essays from AI Human CSV
ai_human_essays = ai_human[ai_human['generated'] == 0.0]

# Filter our AI generated essays from Balanced AI Human Prompts CSV
balanced_generated_essays = balanced_ai_human_prompts[balanced_ai_human_prompts['generated'] == 1]
# Filter our Human essays from Balanced AI Human Prompts CSV
balanced_human_essays = balanced_ai_human_prompts[balanced_ai_human_prompts['generated'] == 0]

# Skip texts that are shorter than min_words (70)
ai_generated_essays = ai_generated_essays[long_mask(ai_generated_essays, min_words=70)]
ai_human_essays = ai_human_essays[long_mask(ai_human_essays, min_words=70)]
balanced_generated_essays = balanced_generated_essays[long_mask(balanced_generated_essays, min_words=70)]
balanced_human_essays = balanced_human_essays[long_mask(balanced_human_essays, min_words=70)]

# Take sample from data
def safe_sample(df, n, seed):
    replace = len(df) < n
    return df.sample(n=n, random_state=seed, replace=replace)

# 375 rows from AI Human CSV + 375 rows from Balanced AI Human Prompts CSV - All AI generated (Total: 750)
generated_sample = pd.concat([
    safe_sample(ai_generated_essays, 375, seed=42),
    safe_sample(balanced_generated_essays, 375, seed=43)
], ignore_index=True)

# 375 rows from AI Human CSV + 375 rows from Balanced AI Human Prompts CSV - All Human written (Total: 750)
human_sample = pd.concat([
    safe_sample(ai_human_essays, 375, seed=44),
    safe_sample(balanced_human_essays, 375, seed=45)
], ignore_index=True)

# Final concatenated 1500 rows (750 human + 750 AI)
final_df = pd.concat([generated_sample, human_sample], ignore_index=True)
final_df = final_df[['text', 'generated']]
final_df['generated'] = final_df['generated'].astype(int)

# Re-validate if any row with less than 70 words length
for index, row in final_df.iterrows():
    if word_count(row['text']) < 70:
        print(f"Row with index {index} has less than 70 words: {row['text']}")


# Shuffle

final_ai = final_df[final_df['generated'] == 1]
final_human = final_df[final_df['generated'] == 0]

final_ai = safe_sample(final_ai, 750, seed=99)
final_human = safe_sample(final_human, 750, seed=100)

final_df = pd.concat([final_ai, final_human], ignore_index=True)

final_df = final_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

# Save

Path('data').mkdir(exist_ok=True)
final_df.to_csv('data/dataset.csv', index=False)
