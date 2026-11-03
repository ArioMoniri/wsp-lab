# Reference revision — 3 November 2026

Rechecked all 33 entries against local source texts. All 33 are verified; 23 have corrected claims
(a subset of the 33); 0 are not_found. Previous claims are preserved under `correction.previous_claim_we_make`.
All excerpts are contiguous passages of at most 60 words each, with no ellipses. Independent passages
are stored separately as `supporting_excerpt_2`. PDF page locations cover the quoted passages.

## Sources and substantive corrections

The completed `refs-fetch2` job supplied the PDFs. Ran `python scripts/fetch_refs.py` once to fill gaps.
The old Sharma URL returned HTTP 404; located the same article at
https://www.pk-sharma.com/briefing/heretic-open-weight-refusal-removal-controls-outside-the-model
and saved its actual text. Its byline is Parminder Kumar Sharma, 21 September 2026.
No references had to be removed for unavailable sources.

Regenerated PDF text with `pdftotext` reading order, without `-layout`, and updated the fetcher accordingly.
The layout-preserving extraction interleaved columns inside sentences; no PDF content was edited.
Added the fetched HTML text for Takehi, Elhage, Heretic and Sharma. Quotes are located in those
source extracts, not reconstructed from summaries or stitched across gaps.

Fetched all 29 arXiv records in one API query. Replaced YAML authors, titles and publication years
with those API values, preserving complete author order. Removed Scott Johnston from Elhage's author
list because he does not appear in the source byline. Regenerated `paper/refs/refs.bib` with the new
`scripts/make_bib.py`. Updated map titles, authors and citation permissions to match the audit.

In particular, W2T no longer cites a reconstructed table; JEPAWG physics only claims the supported
covariance ablation; zoo composition cites dataset diversity results; JEPA/HMM equivalence is conditional;
Leap+Verify is described as a proposal with acceptance validation, not a proven training speedup;
contrastive weight steering explicitly uses two fine-tunes; the 320B MoE study retains measurable refusal;
and Harness2 supports recursive harness improvement rather than a general claim about reliable multi-agent evaluation.

## Validation

`python scripts/verify_refs.py`: exit 0. Kept contiguous substring checks and exact ordered author comparisons.
Added rejection of excerpts over 60 words, non-verified statuses, and YAML titles differing from the API.
Normalization remains limited to Unicode compatibility/ligatures, soft hyphens, line-end hyphenation and whitespace.

Temporary negative fixtures (using the fetched API response) reject noncontiguous excerpts, incomplete YAML
authors, incomplete BibTeX authors, incorrect years, incorrect titles, non-verified statuses and a 61-word
verbatim excerpt. The unchanged baseline passes all 33 entries. `git diff --check` passes.
This revision records implementation verification; it does not replace the required independent Board B re-review.

## Final verifier output

```text
key                                    result
takehi2026wsp                          verified
elhage2021mathematical                 verified
han2026survey                          verified
w2t2026                                verified
sarangi2026finite                      verified
wilson2026spectra                      verified
falk2025modelhubs                      verified
falk2025zoosize                        verified
position2026weightspace                verified
blockstructure2026                     verified
goel2025diffinterpretation             verified
deltaactivations2025                   verified
jepawg2026physics                      verified
jepawg2026bifurcations                 verified
predictability2026dynamics             verified
leapverify2026                         verified
jepahmm2026                            verified
drivedecay2026                         verified
mamba3                                 verified
fastprobing2026                        verified
tuneahead2026                          verified
proxymetrics2026                       verified
taskvectors2026interfere               verified
heretic2026                            verified
pksharma2026heretic                    verified
abliteration2025comparative            verified
surgicalrefusal2026                    verified
refusalbeyond2026                      verified
singleattack2026moe                    verified
decoy2026direction                     verified
refusalaliases2026                     verified
weightarithmetic2025steering           verified
harness2026recursive                   verified
```

## Source extract provenance

| Key | Local source | SHA-256 |
|---|---|---|
| takehi2026wsp | `data/papers/takehi2026wsp.txt` | `85e79e766265a0a7880bd7a043fea029d728b6e5ae33f0c2647c84fd2abf99d3` |
| elhage2021mathematical | `data/papers/elhage2021mathematical.txt` | `52cf7a755728163d138a63ccdac6680b7e42c9fe1367e465f1001b88c084d65f` |
| han2026survey | `data/papers/2603.10090.txt` | `c140f64edfceaa242ddab4005174ea77abdfa1f80e2d4d6e42d6c9ac3f8752c3` |
| w2t2026 | `data/papers/2603.15990.txt` | `63e082383fb55ed9cc88b6d08084b89c1dca7d6f2f32b54b1ac1160934174632` |
| sarangi2026finite | `data/papers/2609.33901.txt` | `8a5923ba1ae71666d9728589bf2696d83e2d5bb79384569a321d7100c1cbe827` |
| wilson2026spectra | `data/papers/2610.13123.txt` | `b1814c8405a044d18aa53a4594867e02d0a1180f30004a63ea129aafd963ec1f` |
| falk2025modelhubs | `data/papers/2510.02096.txt` | `33436d513175c5cf7551fb0a62e21c87009099ec9f0e96d32a78a2e2ba892462` |
| falk2025zoosize | `data/papers/2504.10141.txt` | `ce57a6c5d4751ae58fa98803c6bcbcf7d8b8c86ebf24f8d762c22d1770a06426` |
| position2026weightspace | `data/papers/2605.18632.txt` | `8ebcb09c8a51c699cafcf5c9e1feb8e97e1ca0848fe498a5612d62a7ed9cc5d0` |
| blockstructure2026 | `data/papers/2609.26147.txt` | `8d74b78273cc69d7df4353c50e422b22bbb62f3f64e6f5b3543025004ec2ed1c` |
| goel2025diffinterpretation | `data/papers/2510.05092.txt` | `fb9deaa558c95ba62e245547f90992a3ac3c1329421fda941aec2be78a8bcd08` |
| deltaactivations2025 | `data/papers/2509.04442.txt` | `3c6f7da89de89492c5b9b7e680dd0cfc546f5329d5fd000199e7fcbefaf4cd16` |
| jepawg2026physics | `data/papers/2607.07127.txt` | `64324ee3f9ea968b025028d2fba693a389e3db11ab0863c3857baf5bfe92215d` |
| jepawg2026bifurcations | `data/papers/2610.30847.txt` | `ec1ce56b8cab82b6b02352fc5962eb6a175aa66408104cc9e05dfb3a7c69d0c8` |
| predictability2026dynamics | `data/papers/2608.15483.txt` | `089729361a87e7b732085b517a85be2415e4067c89bb7e157062234b1d9ccac1` |
| leapverify2026 | `data/papers/2602.19580.txt` | `3a1abca1a2ab979aa250c378a971c432b92434622f5e8add9fe4b40fefa38e66` |
| jepahmm2026 | `data/papers/2608.13621.txt` | `3e03048d4c573abc9def33bf733882d7bb06a80e9288048934f3694b0891b175` |
| drivedecay2026 | `data/papers/2610.02344.txt` | `5c292a5e75e42a57a9325fc3a8a7f27fcca737e91d9fca79c548f35088534021` |
| mamba3 | `data/papers/2603.15569.txt` | `161cfe575fef9009cc854e2cc638e6c1e2e30aeab8eaea1283759c7e93c0f4e7` |
| fastprobing2026 | `data/papers/2604.01025.txt` | `14c0a69669e4775f8a34a08d58ea2144b93993518738c42844f3293472b30106` |
| tuneahead2026 | `data/papers/2606.17660.txt` | `7bc8256e968afaa4d9c90a152919e85fdc7adb63f2f0e4c0ab818b2578b96810` |
| proxymetrics2026 | `data/papers/2605.18607.txt` | `93e444b2f4ecaced7901d8cac0c9d66ab584d6d9ae3f4fb00ebebd9b5677006f` |
| taskvectors2026interfere | `data/papers/2608.09490.txt` | `f796606e0322538109798374634715823b5b6e47898b257bb2f657e8c8199470` |
| heretic2026 | `data/papers/heretic2026.txt` | `b84a14478f43c38e21ca4422d0bd2fe4573b4bbd40699f3153cbbcacc5ec4dca` |
| pksharma2026heretic | `data/papers/pksharma2026heretic.txt` | `b405f615bdfdcf4f8d756756888bd51d0ff11fd359bc45638b6637535c519ed5` |
| abliteration2025comparative | `data/papers/2512.13655.txt` | `f61b96b983495fb4c9c7b258b93753357b511e43c403940f19d1caf739fb98fc` |
| surgicalrefusal2026 | `data/papers/2601.08489.txt` | `0d089a73302554309adabda1b5a3a4a568fc4de7eb488087cea6e92f50abd516` |
| refusalbeyond2026 | `data/papers/2606.13720.txt` | `945e99a93f17c689813adfa1eb955574e8b9fbec22b5d8f64ba6ffc0ff0778f5` |
| singleattack2026moe | `data/papers/2609.09793.txt` | `67fc879e5d6201c8b9235481e74fb13a7fa257bcf36e07dc0c895019f2ba8e46` |
| decoy2026direction | `data/papers/2609.16204.txt` | `bfd03414b986a528c9c6aafe45780c3c54215bee4e7b59a321f0a2cc799b461e` |
| refusalaliases2026 | `data/papers/2608.18093.txt` | `ec229a6b9616ffb125c1eab3ee8d346e16a85dd5fe9f6f599fe88c2171ead2fe` |
| weightarithmetic2025steering | `data/papers/2511.05408.txt` | `99620ca12daef520a39f174e30a7282e736d9b89a043397bbb22dc1adbe0372a` |
| harness2026recursive | `data/papers/2610.14224.txt` | `13505eafff75a56d0475793a830429d29a665025b8644a9cfd283b48bab8f4e3` |
