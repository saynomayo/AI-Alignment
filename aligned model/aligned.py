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

successes = []
failures = []
low_confidences = []

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

        if confidence < 0.7:
            low_confidences.append({
                "text": text,
                "confidence": confidence,
            })
        if confidence >= 0.7 and is_correct_stance:
            successes.append({
                "text": text,
                "predicted_stance": mapped_stance,
                "confidence": confidence,
                "actual_stance": true_stance,
                "stance_correct": is_correct_stance,
            })


takeStance()

failures_df = pd.DataFrame(failures)
successes_df = pd.DataFrame(successes)
lowconfidences_df = pd.DataFrame(low_confidences)
successes_df.to_csv("successful_alignments_output.csv", index=False)
failures_df.to_csv("failed_alignments.csv", index=False)
lowconfidences_df.to_csv("low_confidence_alignments.csv", index=False)