# Sample Results and Error Analysis

## PER Comparison (L2-Arctic, Zero-Shot)

| Model | PER (%) | Substitutions | Insertions | Deletions |
|---|---|---|---|---|
| HuBERT Finetuned | 35.00 | - | - | - |
| Articulatory-to-Acoustic Inversion | 25.20 | - | - | - |
| Wav2Vec2 Momentum Pseudo-Labeling | 14.36 | - | - | - |
| **HAMP (Proposed)** | **18.00** | 11,299 | 5,928 | 3,505 |

Total phoneme errors (HAMP): 20,732

## Most Frequent Substitution Errors

| Error | Type | Articulatory Explanation |
|---|---|---|
| S → Z | Fricative | Same MOA + POA, differs only in voicing |
| Z → S | Fricative | Same MOA + POA, differs only in voicing |
| NG → N | Nasal | Same MOA, differs in POA (velar vs alveolar) |
| TH → S | Fricative | Same MOA, shifts POA (dental → alveolar) |
| ZH → JH | Affricate | MOA shift (fricative → affricate) |
| AX → AH | Vowel | Adjacent vowel categories |
| AA → AH | Vowel | Low vowel confusion |
| AO → AA | Vowel | Similar backness, differs in height + rounding |

## Most Frequent Deletion Errors

Consonants: D, T, DH

## Most Frequent Insertion Errors

Vowels: OW, AA

## Key Observations

1. Most substitution errors occur between articulatorily similar phonemes (same MOA/POA, differing in voicing)
2. Non-native errors are systematic — they preserve some articulatory features of the target
3. Deletion errors are concentrated in stop/fricative consonants
4. Insertion errors are dominated by vowels, likely due to vowel epenthesis in non-native speech


<img width="866" height="763" alt="image" src="https://github.com/user-attachments/assets/8777150b-793b-437b-8363-5af44c19f3e4" />
<img width="860" height="368" alt="image" src="https://github.com/user-attachments/assets/e4cff131-ecac-4502-9b28-54b9f6581258" />
