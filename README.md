# Support ticket triage: first version

Sorts ABC support tickets into **Billing, Bug, Account Access, Feature Request or Cancellation**.
Each ticket gets a confidence score and a `needs_human` flag with a reason. This version uses
classic ML (TF-IDF + keyword features + logistic regression). An LLM approach is the planned next
phase.

**In short:** on the 40 labelled tickets the model scores 95% (cross-validated), but most of that
comes from a keyword list tuned to those 40 tickets. **Expect roughly 50–60% on wording it has not
seen**, which matches its 5/10 on the new tickets. It is not accurate enough to route tickets on its
own. What makes it safe to run is the needs-human gate: every wrong answer on the new tickets was
sent to a human, and all three tickets it routed automatically were right.

## How to run

Python 3.10+.

```bash
python -m pip install -r requirements.txt

python -m triage all                 # clean -> evaluate -> train -> predict, in one go
# or step by step:
python -m triage clean               # rebuild dataset/tickets_labelled_clean.csv from raw + audit
python -m triage evaluate            # cross-validation, writes reports/ (~1 min)
python -m triage train               # fit on all 40 tickets, saves models/classifier.joblib
python -m triage predict dataset/tickets_new.csv -o predictions/tickets_new_predictions.csv

python -m pytest                     # 23 tests
```

`predict` accepts any CSV with `ticket_id,text` columns.

## How it works

```
ticket text
  -> redact      card numbers (Luhn-checked), expiry, email, UK phone, street address, postcode
  -> features    word 1-2-grams + character 2-5-grams (TF-IDF) + 5 keyword hit-counts
  -> classifier  logistic regression -> probability for each category
  -> policy      needs_human if confidence < 0.85, or card data / security / outage / legal / too short
  -> output      ticket_id, predicted_category, confidence, second_category, second_confidence,
                 needs_human, flags, reason      (no ticket text, so no personal data in the output)
```

| File | Purpose |
|---|---|
| [triage/redact.py](triage/redact.py) | Removes personal and payment data before anything else touches the text |
| [triage/data.py](triage/data.py) | Builds the cleaned dataset from the raw CSV plus [dataset/label_audit.csv](dataset/label_audit.csv) |
| [triage/models.py](triage/models.py) | The 11 candidate models |
| [triage/lexicon.py](triage/lexicon.py) | Hand-written keyword features |
| [triage/evaluate.py](triage/evaluate.py), [triage/report.py](triage/report.py) | Cross-validation, metrics and the report |
| [triage/policy.py](triage/policy.py) | Needs-human rules and threshold choice |
| [triage/predict.py](triage/predict.py), [triage/cli.py](triage/cli.py) | Training, prediction and the command line |

## Results

Full report: [reports/metrics.md](reports/metrics.md). Every score is **out-of-fold**: a ticket
is only scored by a model that never saw it during training.

### Model comparison

Repeated stratified 5-fold CV, 10 repeats, on the corrected labels.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class (baseline) | 0.15 ± 0.05 | 0.05 |
| Naive Bayes (word counts) | 0.47 ± 0.17 | 0.44 |
| Complement NB (TF-IDF) | 0.41 ± 0.16 | 0.39 |
| Logistic regression (word TF-IDF) | 0.44 ± 0.16 | 0.42 |
| Logistic regression (word + char TF-IDF) | 0.47 ± 0.16 | 0.46 |
| Linear SVM, calibrated (word + char) | 0.43 ± 0.14 | 0.41 |
| Keyword rules only | 0.85 ± 0.11 | 0.81 |
| Naive Bayes + keywords | 0.84 ± 0.11 | 0.81 |
| Linear SVM + keywords | 0.95 ± 0.08 | 0.94 |
| **Logistic regression (word + char) + keywords (chosen)** | **0.95 ± 0.08** | **0.94** |

- **Text models alone barely beat chance.** With about 8 examples per category, almost every
  test ticket uses words the model has never seen.
- **Keyword features add the domain vocabulary that 40 tickets can't teach.** The model still
  learns how much to trust each keyword count.
- **The choice holds up under nested CV.** Choosing the model inside the cross-validation loop
  scores 0.95 ± 0.08, so the result is not just a lucky pick.

### Chosen model, leave-one-out

| Labels | Accuracy | 95% CI | Macro F1 |
|---|---|---|---|
| Corrected | 38/40 = 95% | 83–99% | 0.94 |
| As supplied | 34/40 = 85% | 71–93% | 0.85 |

- **Per category:** Bug, Account Access and Feature Request are 100% correct. Billing and
  Cancellation are each 0.89 and 0.83 F1, because they get confused with each other.
- **Both errors are on tickets marked as ambiguous:**
  - T015, "charged twice … want to cancel and get a refund", is labelled Cancellation and was
    predicted Billing.
  - T040, "downgrade instead of cancelling", is labelled Billing and was predicted Cancellation.
- **On the original labels**, 3 of the 6 errors are the 3 mislabelled tickets, where the model
  gives the correct answer and the supplied label is wrong.

### How much should we trust 95%? Not much.

I wrote the keyword list after reading the 40 labelled tickets. So it partly encodes the answers,
and cross-validation cannot remove that. To measure the effect, the report re-scores the model
without the 38 patterns that each match only a single labelled ticket:

| Keyword features | CV accuracy |
|---|---|
| All 49 patterns | 0.95 |
| Without the 38 single-ticket patterns | **0.575** |
| None | 0.47 |

**The realistic estimate on new wording is about 50–60%.** The 10 new tickets confirm this:
5 of 10 are right by my reading.

### Confidence threshold

| Confidence threshold | Auto-routed | Accuracy on auto-routed |
|---|---|---|
| 0.50 | 100% | 95% |
| 0.70 | 85% | 97% |
| **0.85 (chosen)** | **65%** | **100% (26/26)** |
| 0.95 | 43% | 100% |

- **How it was chosen:** the threshold is the lowest one reaching 98% accuracy on auto-routed
  tickets, and it is never allowed below 0.5. It was picked using the labelled tickets only. See
  [reports/coverage_curve.csv](reports/coverage_curve.csv).
- **Keep it high at launch.** The curve comes from the same optimistic scores, so the threshold
  should stay conservative until real data replaces this estimate.

## Predictions for the 10 new tickets

[predictions/tickets_new_predictions.csv](predictions/tickets_new_predictions.csv). I predicted
once with the frozen model and changed nothing after seeing the results.

| Ticket | Predicted | Conf. | Needs human | Flags | My reading | Right? |
|---|---|---|---|---|---|---|
| N01 billed for a closed location | Billing | 0.68 | yes | | Billing | yes |
| N02 password reset to old email | Account Access | 0.99 | no | | Account Access | yes |
| N03 rota won't load on Firefox | Account Access | 0.38 | yes | | Bug | no |
| N04 swap shifts without approval | Account Access | 0.39 | yes | | Feature Request | no |
| N05 cancel, business sold | Cancellation | 0.86 | no | RETENTION | Cancellation | yes |
| N06 send receipts to … | Account Access | 0.40 | yes | PERSONAL_DATA | Billing | no |
| N07 deleted shifts come back | Account Access | 0.40 | yes | | Bug | no |
| N08 pause instead of cancel | Cancellation | 0.75 | yes | RETENTION | Billing or Cancellation | ok |
| N09 can't create login from invite | Account Access | 0.89 | no | | Account Access (could be Bug) | yes |
| N10 who's approaching overtime | Account Access | 0.37 | yes | | Feature Request | no |

"My reading" is my own judgement, not ground truth.

- **The safety gate worked.** All 7 needs-human tickets were flagged for low confidence. That is
  all 5 wrong answers plus N01 and N08.
- **Nothing wrong was routed automatically.** All 3 auto-routed tickets were correct.
- **It has a fallback bias.** When no keyword fires, the model defaults to Account Access with
  low confidence. N03, N04, N06, N07 and N10 all failed this way. That is the clearest sign it
  needs more vocabulary, and the vocabulary must come from new labelled data, not these tickets.

## What I noticed about the data

The full list, with reasons, is in [dataset/label_audit.csv](dataset/label_audit.csv). The raw
file is never edited; the cleaned file is generated from the raw file plus the audit.

1. **Three tickets are mislabelled (7.5%):**
   - T007, "login link keeps expiring", is labelled Billing but should be **Account Access**.
   - T019, "add a feature to cancel a shift swap", is labelled Cancellation but should be
     **Feature Request**. The word "cancel" was taken to mean leaving.
   - T033, "PDF export cuts off a column", is labelled Feature Request but should be **Bug**,
     because something that exists is broken.
2. **Five tickets have two intents.** T015, T006, T039, T030 and T040 keep their labels but are
   marked ambiguous, with a second label recorded. The one-category taxonomy can't represent them.
   These caused both cross-validation errors, and N08 and N09 in the new set are similar.
3. **Some tickets contain personal and payment data:**
   - T005 has a full card number and expiry. If this were real data it would be a PCI DSS
     problem: card numbers must not sit in a ticket system, a log or a third-party API.
   - T012 and N06 have an email address and phone number.
   - T027 has a postal address.

   All of it is redacted before training and prediction. The raw values here look like dummy
   values (4111… is a standard test card and 07700 900xxx is a reserved fictional range), which
   is why the raw file is committed.
4. **Some tickets matter more than their category suggests:**
   - T024, "staff can see each other's pay rates", is a data-exposure incident, not an ordinary
     bug.
   - T030, "nobody at our Bristol site can log in", is an outage.

   Both are escalated by the policy whatever their category.
5. **There is no "Other" or "Spam" category**, so anything off-topic is forced into one of the
   five. The needs-human gate is the only protection against that.
6. **The dataset is tiny.** There are 40 rows, 6–9 per category, so the 95% confidence interval
   on accuracy is about ±10–15 points.

## How it was measured and kept honest

- **Cross-validation.** Model comparison uses repeated stratified 5-fold CV, 10 repeats. Error
  analysis uses leave-one-out CV. Model choice is checked with nested CV. The vectorisers sit
  inside each pipeline, so they are re-fitted on every training fold.
- **Two sets of labels.** Results are reported against both the corrected and the supplied labels.
- **The new tickets are a blind test set.** While building this, I found that six keyword
  patterns and one retention rule had been influenced by reading `tickets_new.csv` early on.
  They were removed in commit `953701d`.
  - **The rule now:** every keyword pattern must match at least one labelled ticket, and
    [tests/test_lexicon.py](tests/test_lexicon.py) enforces it.
  - **Why some generic words are missing:** words like "error" and "broken" were removed by the
    same rule, though they matched none of the 50 tickets. They should come back from future
    labelled data.
- **Correction to a commit message.** The message on commit `aa4dd96` says 6 of the 7 flagged
  tickets were flagged for low confidence. In fact all 7 were.

## What I'd do before putting this live

1. **Get real labels.** Label 300–500 recent tickets, two people per ticket, using a written
   guide with tie-break rules. Examples of rules:
   - "Asking for something new is a Feature Request even if it says cancel."
   - "Intent to leave is Cancellation."
   - "Something that exists and doesn't work is a Bug."

   Measure how often the two labellers agree. That agreement is the realistic ceiling for any
   model.
2. **Fix the taxonomy.** Allow a secondary category, or route multi-intent tickets to a human.
   Add Other/Spam. Treat security and outage as priorities separate from category.
3. **Run in shadow mode for 2–4 weeks.** The model predicts but humans still sort. Track
   agreement per category, the needs-human rate, and accuracy on auto-routed tickets. Only then
   switch on auto-routing, starting with high-confidence tickets.
4. **Grow the keyword list and retrain from real labelled tickets.** Set the threshold again on
   a held-out set that was never used for tuning.
5. **Data protection.** Redact before storage and logging, not only before the model. Add an
   automated reply telling customers not to send card details, and purge card data that arrives.
   Agree a retention policy. Check redaction recall on real tickets, because regexes miss things.
6. **Operations:**
   - Monitor the category mix (drift), the human override rate, and the confidence distribution.
   - Alert when overrides spike.
   - Version the model and keep a rollback path.
   - If the model fails, send every ticket to the human queue. Never drop tickets.
7. **Feedback loop.** Every human correction becomes a labelled example. Retrain on a schedule
   and re-run the same evaluation before each release.

## Next phase: LLM approach

A free-tier API or a local open-source model would bring the general vocabulary this model lacks.
On N03, N04, N07 and N10 it would have understood the meaning without keywords. Plan:

- Use the same `predict_proba`-style interface, the same redaction step before the call, and the
  same needs-human policy.
- Score it with the same CV on the same 40 tickets, scored once on the blind set.
- Compare it with this model on accuracy, cost per ticket, latency and data-handling
  requirements.
- A likely end state is a hybrid: the cheap classic model handles high-confidence tickets, and
  the LLM handles the rest before anything reaches a human.

## Repository layout

```
dataset/      raw CSVs (untouched), label_audit.csv, generated tickets_labelled_clean.csv
triage/       package: redact, data, lexicon, models, evaluate, report, policy, predict, cli
tests/        redaction, data cleaning, leakage guard, policy, prediction output
reports/      metrics.md + CSVs (comparison, out-of-fold predictions, errors, coverage, confusion)
models/       classifier.joblib (trained on all 40 labelled tickets)
predictions/  tickets_new_predictions.csv
```
