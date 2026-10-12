# Bibliography check B: REFS of fragments/L07.tex and fragments/L08.tex

Checked on 2026-10-12 by an independent fact-checker. No existing file was edited; raw data and scripts are in the scratchpad folder `.../scratchpad/bibcheckB/` (arXiv batch XML, Crossref JSON, OpenReview JSON, small parsers).

## Summary

63 distinct keys checked: **59 OK, 3 FIX, 1 UNVERIFIED.**

- No reference is fabricated or unfindable. Every 2026 source exists at the stated URL/id with the stated title, authors (or author count) and date: lu2026assistant, cunningham2026, aisi2026bpj, khodayari2026, betley2026nature, anthropic2026persona, brunner2026, sewani2026, moonshot2026, openai2026dns, openai2026math (also betleyweird2025, chismon2025).
- Every printed arXiv author list (names, initials, order) and every arXiv id and title was compared with the arXiv record by script, and every "et al." count (34, 7, 9, 22, 4, 11, 29, 16, 14, 43) with the arXiv, Nature or NeurIPS record: no discrepancy beyond the three FIX rows below.
- **FIX gwern2023** (most substantive): the URL is the post, not the comment (21 comments by Gwern among 182), and "February 2023 (in existence by 19 February 2023)" is imprecise. The comment the lecture means is the top-level comment `commentId=AAC8jKeDp6xqsZK2K`, first posted 2023-02-17 02:06 UTC (since edited).
- **FIX aghajanyan2021**: the entry cites the ACL pages (7319--7328) but follows the arXiv author order; the ACL version lists Aghajanyan, Gupta, Zettlemoyer.
- **FIX tan2025**: accent, Rich\'e (all else correct).
- **UNVERIFIED roose2023**: nytimes.com cannot be fetched with the available tools; headline, byline, URL and date agree with third-party listings; the title of the separately published transcript (in the parenthesis) is unconfirmed.
- Venue claims checked and confirmed: Soligo (ICML 2025 Workshop on Actionable Interpretability: the paper is on the workshop's ICML page), Tan and Wang (ICLR 2026 posters on OpenReview), Kumar (COLM 2024), Betley (ICML 2025, PMLR 267:4043--4068), Anil (NeurIPS 2024, vol. 37), Perez and Ribeiro (ML Safety Workshop, NeurIPS 2022, from the arXiv comment).
- Merge check (read-only replication of tools/merge_fragments.py, 169 entries): no author-year label collisions; betley2025 comes from L7 (label 2025a, with betleyweird2025 = 2025b), anil2024, hughes2024, sharma2025, turner2025, openai2026dns (2026e) and openai2026math (2026f) come from tools/refs_override.tex.

Scope: L07 has 31 entries; baker2025, korbak2025 and rafailov2023 are in L06's REFS and were skipped. L08 has 39 entries; anil2024, betley2025, hughes2024, turner2025 are shared with L07, and sharma2025 (also in L06) is checked in its override version as instructed. No key of L07/L08 is in tools/references.base.tex. 28 + 35 = 63 keys.

## Method and limits

- arXiv: records from the arXiv export API (the data behind the abstract pages) for all 36 ids, plus abstract pages fetched for 2506.11618, 2602.15001, 2601.10387, 2604.27202, 2601.04603, 2512.09742; HTML version of 2602.15001 for affiliations.
- Journals/proceedings: Crossref API (Nature, IEEE, NIST, Wiley, Taylor & Francis, ACM, ACL Anthology, NeurIPS, Cambridge), PMLR, JMLR, ACL Anthology, NeurIPS and ICML pages, SLMath/MSRI contents page; OpenReview API for venues.
- LessWrong posts and comments: LessWrong's own GraphQL API. Timestamps are UTC; the dates in the entries are the UTC dates. In US time zones the Nardo post and the Gwern comment fall one day earlier, and so does the Hubinger post except in US Eastern time.
- Blogs and reports: page HTML read directly (title, JSON-LD author and date): Anthropic (3 pages), OpenAI alignment report, Google, Forcepoint, NCSC, Embrace The Red, Simon Willison, OWASP, Hugging Face, Moonshot tech blog, GitHub API for openai/math and MoonshotAI/Kimi-K3.
- Not reachable: nytimes.com (the fetch tool refuses the host; not worked around), openai.com pages (HTTP 403; OpenAI's own RSS feed was used for the post title and date), web.archive.org (refused), nature.com (cookie redirect; Crossref used), msri.org mirror (connection refused; library.slmath.org used).
- "Label OK" means: first author(s) as in the entry, "et~al." only for three or more authors, year = the year printed in the entry (for entries with a later venue year in parentheses the label keeps the first year, as in the entry).

## Table

Order: keys shared by L07 and L08 (override rules), then sharma2025, then L07-only keys, then L08-only keys, each alphabetical.

| key | status | what is wrong or unconfirmed | corrected entry (FIX rows only) |
|---|---|---|---|
| anil2024 | OK | Override version. NeurIPS 2024 proceedings page and Crossref (DOI 10.52202/079017-4121): 34 authors; first seven are Anil, Durmus, Panickssery, Sharma, Benton, Kundu, Batson, in this order; title "Many-shot Jailbreaking"; Advances in Neural Information Processing Systems 37 (pp. 129696--129742, optional). OpenReview: NeurIPS 2024 poster. Anthropic page dated 2 Apr 2024 (page metadata). Label OK. | |
| betley2025 | OK | L7 version (the merge keeps it because L7 is the first definition; label 2025a). arXiv 2502.17424 and the PMLR page: 8 authors in the printed order (Tan = Daniel Chee Hian Tan); title exact; PMLR 267:4043--4068 (citation metadata), 42nd ICML; OpenReview: ICML 2025 oral. Labels 2025a, 2025b (betleyweird2025) and 2026 (Nature) are consistent. | |
| hughes2024 | OK | Override version. arXiv 2412.03556 (v2): the 10 authors in the printed order; title exact; v1 4 Dec 2024. Optional: the NeurIPS 2025 poster version has 11 authors (A. Somani added); the entry correctly follows the arXiv record. Label OK. | |
| turner2025 | OK | Override version. arXiv 2506.11613: 5 authors in the printed order; title exact; v1 13 Jun 2025. Optional venues (OpenReview): ICML 2025 R2-FM Workshop poster; Mech Interp Workshop, NeurIPS 2025, poster. Label OK. | |
| sharma2025 | OK | Override version (also in L06; the override replaces L06's text). arXiv 2501.18837: 43 authors, first three Sharma, Tong, Mu as printed; title exact; v1 31 Jan 2025. Label OK. | |
| anthropic2025vend | OK | Page title exactly "Project Vend: Can Claude run a small shop? (And why does that matter?)"; published 2025-06-27 (page metadata); Andon Labs named as partner; URL resolves. Label OK. | |
| anthropic2026persona | OK | alignment.anthropic.com/2026/psm: title "The Persona Selection Model: Why AI Assistants might Behave like Humans"; authors Sam Marks, Jack Lindsey, Christopher Olah; "February 23, 2026". The short version at anthropic.com/research/persona-selection-model is titled "The persona selection model", published 2026-02-23 (no authors printed there). Label OK. | |
| betley2026nature | OK | Crossref/Nature: 9 authors, first J. Betley (the Nature list adds M. Srivastava to the arXiv list); title exact; Nature 649(8097):584--589; published online 14 Jan 2026; DOI matches. Label OK. | |
| betleyweird2025 | OK | arXiv 2512.09742 (abstract page fetched): 7 authors, first J. Betley; title exact; v1 10 Dec 2025. Label 2025b OK. | |
| chen2025persona | OK | arXiv 2507.21509: 5 authors in the printed order; title exact; v1 29 Jul 2025 (the ICLR 2026 submission was rejected; the entry claims no venue). Label OK. | |
| gwern2023 | FIX | The URL is the post's, not the comment's, so a reader cannot find the comment (182 comments, 21 by Gwern). The comment that matches the lecture's use (a GPT-4 model fine-tuned on dialogues, not RLHF) is Gwern's top-level comment `commentId=AAC8jKeDp6xqsZK2K`, first posted 2023-02-17 02:06 UTC (LessWrong API; 16 Feb in US time zones) and edited since (it contains "EDIT:"). "February 2023 (in existence by 19 February 2023)" is true but imprecise. The same hedge "in existence by 19 February 2023" appears in lectures/lecture07.tex (line 130) and fragments/L07_report.md (line 25). Label OK. | ``` \bibitem[Gwern(2023)]{gwern2023} Gwern. Comment on ``Bing Chat is blatantly, aggressively misaligned''. LessWrong, 17 February 2023. \url{https://www.lesswrong.com/posts/jtoPawEhLNXNxvgTT/bing-chat-is-blatantly-aggressively-misaligned?commentId=AAC8jKeDp6xqsZK2K} ``` |
| hubinger2023 | OK | LessWrong API: title exact; author evhub (full name Evan Hubinger), no coauthors; posted 2023-02-15 05:29 UTC (14 Feb in US Pacific, Mountain and Central time); URL matches. Label OK. | |
| janus2022 | OK | LessWrong API: "Simulators" by janus, posted 2022-09-02 12:45 UTC; URL matches. Label OK. | |
| korbak2022 | OK | arXiv 2205.11275: 3 authors (C. L. Buckley = Christopher L. Buckley); title exact; ACL Anthology/Crossref: Findings of EMNLP 2022 (pp. 1083--1091, optional). Label OK. | |
| lu2026assistant | OK | arXiv 2601.10387 (abstract page fetched): 5 authors in the printed order; title exact; v1 15 Jan 2026. Label OK. | |
| macdiarmid2025 | OK | arXiv 2511.18397: 22 authors, first M. MacDiarmid; title exact; v1 23 Nov 2025. Label OK. | |
| mori2012 | OK | Crossref/IEEE: IEEE Robotics & Automation Magazine 19(2):98--100, June 2012 (DOI 10.1109/MRA.2012.2192811), translated by Karl F. MacDorman and Norri Kageki (also credited on the IEEE Spectrum reprint). Original: Energy, vol. 7, no. 4, pp. 33--35, 1970 (credit line on the IEEE Spectrum page). Label year 2012 = translation. | |
| nardo2023 | OK | LessWrong API: "The Waluigi Effect (mega-post)" by Cleo Nardo, posted 2023-03-03 03:22 UTC (2 Mar in US time zones); no coauthors; URL matches. Label OK. | |
| openai2026dns | OK | Override version (label 2026e). Page title "An agent used DNS to reach an external chatbot" on the OpenAI Alignment Research Blog, section "Misalignment Reports and Notices"; "Sample: Sep 20, 2026; Discovery: Sep 20, 2026; Report updated: Sep 25, 2026" (page text); no author printed. Label OK. | |
| roose2023 | UNVERIFIED | nytimes.com is refused by the fetch tool, so the NYT page was not read. Headline "A Conversation With Bing's Chatbot Left Me Deeply Unsettled", byline Kevin Roose, the exact URL (href on a Benton.org listing) and the date 16 Feb 2023 (updated 17 Feb) agree with third-party listings and search summaries. Not confirmed: the title of the separately published transcript in the parenthesis (a search summary gave "Bing's A.I. Chat Reveals Its Feelings: 'I Want to Be Alive.'" with an emoji; the entry has different capitalization). Suggest checking that one phrase by hand or dropping the parenthesis. | |
| schaeffer2025 | OK | arXiv 2502.17578: 10 authors in the printed order (J. Kazdan = Joshua Kazdan); title exact; v1 24 Feb 2025. Optional: ICML 2025 oral (OpenReview). Label OK. | |
| shanahan2023 | OK | Crossref: Nature 623(7987):493--498, online 8 Nov 2023, issue date 16 Nov 2023; 3 authors Murray Shanahan, Kyle McDonell, Laria Reynolds; title exact; DOI matches. Label OK. | |
| soligo2025 | OK | arXiv 2506.11618 (abstract page fetched): 4 authors, first A. Soligo; title exact; v1 13 Jun 2025. The workshop claim is confirmed: the ICML 2025 "Actionable Interpretability" workshop page (icml.cc/virtual/2025/workshop/39962) lists the paper (the arXiv page has no comment field). Optional: OpenReview also shows Mech Interp Workshop, NeurIPS 2025, spotlight. Label OK. | |
| tan2025 | FIX | Accent only: arXiv and OpenReview print Maxime Rich\'e (Riché). All else is correct: 7 authors in the printed order, arXiv 2510.04340, title of arXiv v4, ICLR 2026 poster (OpenReview). Note, no change needed: the ICLR camera-ready title is "...can reduce trait expression at test-time" (arXiv v1 had it too); the entry follows the current arXiv title. Label OK. | `\bibitem[Tan et~al.(2025)]{tan2025} D.~Tan, A.~Woodruff, N.~Warncke, A.~Jose, M.~Rich\'e, D.~D. Africa, M.~Taylor. Inoculation prompting: eliciting traits from LLMs during training can suppress them at test-time. arXiv:2510.04340, 2025 (ICLR 2026).` |
| tierney1986 | OK | Crossref: JASA 81(393):82--86, March 1986; authors Luke Tierney, Joseph B. Kadane; title exact. Label OK. | |
| vandervaart1998 | OK | Crossref (DOI 10.1017/CBO9780511802256): Cambridge University Press, issued 13 Oct 1998 (online edition 2012); single author A. W. van der Vaart. Label OK. | |
| wang2025persona | OK | arXiv 2506.19823: 11 authors in the printed order (T. Dupr\'e la Tour accent as on arXiv); title exact; ICLR 2026 poster (OpenReview and the ICLR proceedings page). Label OK. | |
| wichers2025 | OK | arXiv 2510.05024: 11 authors, first N. Wichers; title exact. The two inoculation-prompting ids are not swapped (04340 = Tan, 05024 = Wichers). The ICLR 2026 submission was rejected and the entry claims no venue, correctly. Label OK. | |
| wolf2023 | OK | arXiv 2304.11082: 5 authors in the printed order; title exact; v1 19 Apr 2023. Optional: ICML 2024 poster (OpenReview). Label OK. | |
| aghajanyan2021 | FIX | Author order. ACL Anthology and Crossref (DOI 10.18653/v1/2021.acl-long.568; ACL-IJCNLP 2021, Long Papers, pp. 7319--7328, Aug 2021) list Aghajanyan, Gupta, Zettlemoyer; the entry follows the arXiv order (Aghajanyan, Zettlemoyer, Gupta). Since the entry cites the ACL pages, the ACL order is the right one. Title, pages, arXiv id are correct. Label OK. | `\bibitem[Aghajanyan et~al.(2021)]{aghajanyan2021} A.~Aghajanyan, S.~Gupta, L.~Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. \emph{ACL} 2021, 7319--7328; arXiv:2012.13255.` |
| aisi2026bpj | OK | arXiv 2602.15001 (abstract page and HTML fetched): 6 authors in the printed order; title exact; v1 16 Feb 2026 (v2 18 Feb). Affiliations on the HTML version: UK AI Security Institute (all six), OATML Oxford (X. Davies, Y. Gal). Label OK. | |
| ball1997 | OK | SLMath/MSRI contents page: "An Elementary Introduction to Modern Convex Geometry, by Keith M. Ball, 1-58" in Flavors of Geometry (MSRI Publications 31); Crossref: Cambridge University Press, 28 Sep 1997. Label OK. | |
| bazinska2025 | OK | arXiv 2510.22620: 7 authors in the printed order; title exact; v1 26 Oct 2025 (v2 24 Feb 2026). Label OK. | |
| brunner2026 | OK | blog.google page: title exact; authors Thomas Brunner, Yu-Han Liu, Moni Pande; published 2026-04-23. The post first appeared on the Google Security Blog (security.googleblog.com/2026/04/..., now a 301 redirect to the blog.google URL in the entry), so the venue name is right. Label OK. | |
| cai2013 | OK | JMLR page metadata: vol. 14, issue 57, pp. 1837--1864, 2013; 3 authors; title exact. Label OK. | |
| chismon2025 | OK | NCSC page: title exact; JSON-LD author Dave Chismon ("Written by Dave Chismon, CTO for Architecture"); published 8 Dec 2025. The post's sign-off reads "David C, NCSC Technical Director for Platforms Research", so the surname appears only in the byline box. Label OK. | |
| cunningham2026 | OK | arXiv 2601.04603 (abstract page fetched): 29 authors; first three H. Cunningham, J. Wei, Z. Wang; title exact; v1 8 Jan 2026. Label OK. | |
| dasgupta2003 | OK | Crossref: Random Structures & Algorithms 22(1):60--65 (online 25 Nov 2002, issue Jan 2003); 2 authors; title exact. Label OK. | |
| debenedetti2025 | OK | arXiv 2503.18813: 10 authors in the printed order (Tram\`er accent as on arXiv); title exact; v1 24 Mar 2025. Label OK. | |
| elhage2022 | OK | arXiv 2209.10652: 16 authors; first three Elhage, Hume, Olsson; title exact; v1 21 Sep 2022. Label OK. | |
| ethayarajh2019 | OK | Single author; title exact; EMNLP-IJCNLP 2019, pp. 55--65 (Crossref; arXiv comment "Accepted to EMNLP 2019"). Label OK. | |
| greshake2023 | OK | arXiv 2302.12173: 6 authors in the printed order; title exact; Crossref: Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec '23), pp. 79--90, Nov 2023. Label OK. | |
| hu2021lora | OK | arXiv 2106.09685 (v2): 8 authors in the printed order; title exact; ICLR 2022 (iclr.cc poster page). Label year 2022 (ICLR) with key year 2021 is fine. | |
| kalajdzievski2023 | OK | arXiv 2312.03732: single author; title exact; submitted 28 Nov 2023 (id 2312 because announced in December). Label OK. | |
| khodayari2026 | OK | arXiv 2604.27202 (abstract page fetched): 4 authors in the printed order; title exact; v1 29 Apr 2026. Label OK. | |
| kumar2023 | OK | arXiv 2309.02705: 6 authors in the printed order; title exact; arXiv comment and OpenReview: COLM 2024. Label OK (year = arXiv year as in the entry). | |
| li2023deepinception | OK | arXiv 2311.03191: 6 authors in the printed order; title exact; v1 6 Nov 2023. Label OK. | |
| moonshot2026 | OK | Hugging Face page moonshotai/Kimi-K3 exists (README title "Kimi K3"; HF API: repo created 2026-06-13, last modified 2026-09-02). The card prints no date, but its benchmark notes are dated 9--23 Jul 2026, the Moonshot tech blog (kimi.com/blog/kimi-k3) says the weights "will be released by July 27, 2026", and the GitHub repo MoonshotAI/Kimi-K3 was created 2026-07-27, so "July 2026" is supported. Label OK. | |
| nasr2025 | OK | arXiv 2510.09023: 14 authors; first three Nasr, Carlini, Sitawarin; title exact; v1 10 Oct 2025. Label OK. | |
| openai2025oss | OK | arXiv 2508.10925: author "OpenAI" (followed by a long contributor list); title "gpt-oss-120b & gpt-oss-20b Model Card"; v1 8 Aug 2025. Label OK. | |
| openai2026math | OK | Override version (label 2026f). OpenAI pages return HTTP 403, but OpenAI's own RSS feed lists "Sharing AI progress in mathematics", link openai.com/index/sharing-ai-progress-in-mathematics, pubDate 6 Oct 2026. GitHub API: openai/math created 2026-10-06 (initial commit 21:58 UTC); README: manuscripts and proof artifacts "produced by an internal OpenAI model". Optional: add the post URL next to the repository URL. Label OK. | |
| owasp2025 | OK | Page title "LLM01:2025 Prompt Injection" on the OWASP Gen AI Security Project site (modified 2025-04-17); the parent document is titled "OWASP Top 10 for LLM Applications 2025" (resource page, Nov 2024). Label OK. | |
| perez2022 | OK | arXiv 2211.09527: 2 authors (Fábio Perez, Ian Ribeiro); title exact; arXiv comment: "ML Safety Workshop NeurIPS 2022". Label OK. | |
| rehberger2023 | OK | Page title "Image to Prompt Injection with Google Bard"; "Posted on Jul 14, 2023"; site "Embrace The Red". The page byline is the handle "wunderwuzzi" (an embedded tweet is attributed to "Johann Rehberger (@wunderwuzzi23)"), so the real name rests on that match. Label OK. | |
| robey2023 | OK | arXiv 2310.03684: 4 authors in the printed order; title exact; v1 5 Oct 2023. Label OK. | |
| sewani2026 | OK | Forcepoint page: headline "10 Indirect Prompt Injection Payloads Caught in the Wild" (the HTML title differs: "Indirect Prompt Injection in the Wild: X-Labs Finds 10 IPI Payloads"); author Mayur Sewani; published 22 Apr 2026. The URL redirects to the canonical .../x-labs/indirect-prompt-injection-payloads and works. Label OK. | |
| shen2024dan | OK | arXiv 2308.03825: 5 authors in the printed order; title exact; Crossref: ACM CCS 2024, pp. 1671--1685, DOI 10.1145/3658644.3670388. Label OK. | |
| vassilev2025 | OK | NIST page: "Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations"; 6 authors in the printed order (Vassilev, Oprea, Fordyce, Anderson, Davies, Hamin); March 2025; NIST AI 100-2e2025; DOI matches. Label OK. | |
| wallace2024 | OK | arXiv 2404.13208: 6 authors in the printed order; title exact; v1 19 Apr 2024. Label OK. | |
| wei2023jailbroken | OK | arXiv 2307.02483: 3 authors; title exact; OpenReview: NeurIPS 2023 (oral). Label OK. | |
| welch1974 | OK | Crossref: IEEE Trans. Inform. Theory 20(3):397--399, May 1974; sole author L. R. Welch; the journal adds "(Corresp.)" to the title. Label OK. | |
| willison2023 | OK | Page title "Multi-modal prompt injection image attacks against GPT-4V"; author Simon Willison; "14th October 2023". Label OK. | |
| zou2023 | OK | arXiv 2307.15043: 6 authors in the printed order; title exact; v1 27 Jul 2023. Label OK. | |

## Ready-to-paste corrected entries

```tex
\bibitem[Gwern(2023)]{gwern2023} Gwern. Comment on ``Bing Chat is blatantly, aggressively misaligned''. LessWrong, 17 February 2023. \url{https://www.lesswrong.com/posts/jtoPawEhLNXNxvgTT/bing-chat-is-blatantly-aggressively-misaligned?commentId=AAC8jKeDp6xqsZK2K}
\bibitem[Aghajanyan et~al.(2021)]{aghajanyan2021} A.~Aghajanyan, S.~Gupta, L.~Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. \emph{ACL} 2021, 7319--7328; arXiv:2012.13255.
\bibitem[Tan et~al.(2025)]{tan2025} D.~Tan, A.~Woodruff, N.~Warncke, A.~Jose, M.~Rich\'e, D.~D. Africa, M.~Taylor. Inoculation prompting: eliciting traits from LLMs during training can suppress them at test-time. arXiv:2510.04340, 2025 (ICLR 2026).
```

Where each belongs: gwern2023 and tan2025 are defined in fragments/L07.tex (REFS), aghajanyan2021 in fragments/L08.tex (REFS). They can be edited there, or added to tools/refs_override.tex (the merge lets an override replace any key).

## Side observations (outside the requested scope, not verified further)

- tools/references.base.tex, `openai2026unit` (OpenAI(2026a), "Model disproves discrete geometry conjecture", 20 May 2026): OpenAI's RSS feed lists that item (same URL slug, same date) as "An OpenAI model has disproved a central conjecture in discrete geometry". The base title matches the URL slug, not the feed title; the page itself returned 403, so the correct title is unconfirmed. The feed dates for `openai2026ns` (8 Sep 2026; entry says "September 2026") and `openai2026ra` (6 Sep 2026) agree with the base entries.
- lectures/lecture07.tex line 130 and fragments/L07_report.md line 25 say the Gwern comment was "in existence by 19 February 2023"; if the entry is changed, "posted on 17 February 2023" is accurate (UTC). The generated references.tex still carries the old entry until the merge is re-run.
- Optional venue upgrades verified on OpenReview (not errors; the entries make no venue claim): schaeffer2025 (ICML 2025 oral), wolf2023 (ICML 2024 poster), hughes2024 (NeurIPS 2025 poster, 11 authors), turner2025 (ICML 2025 R2-FM Workshop; NeurIPS 2025 Mech Interp Workshop), soligo2025 (also NeurIPS 2025 Mech Interp Workshop, spotlight).
