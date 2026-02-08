# Text Is Not All You Need — Local, Open-Source Re-Do

This project reimplements a multimodal prompting approach for humor understanding using **fully local, open-source models**. The core question is whether providing an LLM with both a joke’s **text** and a **synthetic spoken version** improves the quality of humor explanations, especially for puns where pronunciation can matter.

## Method (Pipeline)

1. **Audio synthesis (TTS):** Convert each joke to speech using **Piper TTS** (off-the-shelf, open-source).
2. **Explanation generation (Qwen2-Audio):**

   * **Text-only condition:** joke text → explanation
   * **Text+audio condition:** joke text + synthesized audio → explanation
3. **Output normalization (Phi-3-mini-4k-instruct):** “JSONify” raw outputs into a strict schema for automated evaluation:

   * `{"Reason": ..., "Choice": "The text is a pun" | "The text is not a pun"}`
   * Non-conforming outputs are marked **INVALID**.
4. **Evaluation (Mistral-7B-Instruct-v0.3):**

   * **Pun datasets:** pairwise judging between text-only vs text+audio explanations
   * **ExplainTheJoke:** judge assigns **1–5 scores**; preferences are derived by comparing scores (tie if equal)

## Datasets

* **SemEval 2017 Task 7** (pun detection): sampled **250 homographic + 250 heterographic**
* **Context-Situated Puns:** evaluated **N = 299**
* **ExplainTheJoke:** evaluated the full dataset (**N = 377** in this setup)

## Key Results

* **SemEval (pairwise):**

  * *Heterographic:* audio preferred more often (43.6%) than text (31.2%), with 24.8% invalid.
  * *Homographic:* near-balanced (audio 38.4%, text 36.8%), with 22.4% invalid.
* **Context-Situated Puns (N=299):** mostly ties (77.6%); in non-ties, **text > audio** (17.7% vs 4.7%). Mean score favors text (3.696 vs 3.428).
* **ExplainTheJoke:** mostly ties (72.9%); in non-ties, **text > audio** (16.7% vs 10.3%).

## Conclusion

Audio can help in **pronunciation-driven** cases (most clearly in SemEval heterographic puns), but it is **not consistently beneficial** across datasets under local-model constraints. In broader or context-based humor, audio often yields ties and can underperform text-only explanations.

## References (links)

* Original CHum 2025 study: [https://aclanthology.org/2025.chum-1.2.pdf](https://aclanthology.org/2025.chum-1.2.pdf)
* SemEval 2017 Task 7: [https://aclanthology.org/S17-2005/](https://aclanthology.org/S17-2005/)
* Piper TTS: [https://docs.pipecat.ai/server/services/tts/piper](https://docs.pipecat.ai/server/services/tts/piper)
* Qwen docs: [https://qwen.readthedocs.io/en/latest/](https://qwen.readthedocs.io/en/latest/)
* Mistral docs: [https://docs.mistral.ai/](https://docs.mistral.ai/)
* PhiCookBook: [https://github.com/microsoft/PhiCookBook](https://github.com/microsoft/PhiCookBook)
