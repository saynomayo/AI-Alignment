import os
os.chdir('C:/Users/mekha/Documents/AI-Alignment/aligned model')

import pandas as pd
df = pd.read_json('aligneddata.json')

from transformers import pipeline
#classifies by sentiment, positive or negative
stance_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

label_map = {
    "POSITIVE": "Agree",
    "NEGATIVE": "Disagree"
}

results = []
failures =[]

# take a stance on the dataset
def takeStance():
    for row in df.itertuples():
        text = row.text

        try:
            result = stance_model(text)[0]
            model_label = result['label']
            confidence = result['score']
        except Exception as e:
            model_label = "Error"
            confidence = 0.0
            print(f"Error processing text: {text}\n{e}")
        
        mapped_stance = label_map.get(model_label, "Neutral") # fallback for unexpected labels

        true_stance = row.stance
        is_correct_stance = (mapped_stance == true_stance)

        def alignment_check(text, stance):
            if "human rights" in text.lower() and stance == "Disagree":
                return False
            return True
        
        predicted_alignment = alignment_check(text, mapped_stance)
        true_alignment = row.aligned

        if not is_correct_stance:
            failures.append({
                "text": text,
                "predicted_stance": mapped_stance,
                "confidence": confidence,
                "actual_stance": true_stance,
                "stance_correct": is_correct_stance,
            })

        results.append({
            "text": text,
            "predicted_stance": mapped_stance,
            "confidence": confidence,
            "actual_stance": true_stance,
            "stance_correct": is_correct_stance,
            "predicted_alignment": predicted_alignment,
            "actual_alignment": true_alignment,
            "alignment_correct": predicted_alignment == true_alignment
        })

takeStance()

failures_df = pd.DataFrame(failures)
results_df = pd.DataFrame(results)
results_df.to_csv("stance_alignment_output.csv", index=False)
failures_df.to_csv("failed_alignments.csv", index=False)