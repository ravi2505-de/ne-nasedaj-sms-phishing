---
title: Smishi / Ne Nasedaj
emoji: 🎣
colorFrom: red
colorTo: yellow
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
license: mit
tags:
  - text-classification
  - phishing-detection
  - serbian
  - bosnian
  - croatian
  - sms-security
  - cybersecurity
  - south-slavic-nlp
---
# 🎣 Smishi — Ne Nasedaj

**SMS Phishing Detector for Serbian, Bosnian, Croatian & Montenegrin**

*"Ne Nasedaj" — Don't Fall For It.*

## What is this?

Smishi is an open-source SMS phishing (smishing) detector purpose-built for South Slavic languages — Serbian, Bosnian, Croatian, and Montenegrin (SBCM).

Paste an SMS message. The model tells you whether it looks like a scam, in either Serbian or English (language toggle in the UI).

No app. No login. No data stored. Just paste and check.

## Why does this exist?

Smishing is one of the fastest-growing cyber threats in the Western Balkans. Large-scale campaigns impersonating institutions like Pošta Srbije have targeted Serbian citizens for financial fraud and credential theft. Phishing attacks globally have surged sharply since 2022, amplified by AI tools that make localized, convincing fake messages trivially easy to produce at scale.

Yet no publicly available smishing detection tool supports SBCM languages. Generic multilingual models struggle with:

- **Rich morphological case inflection** — *nagrada / nagradu / nagradi* (prize, accusative, dative) are the same attack keyword in three forms
- **Cyrillic ↔ Latin script duality** — identical messages can appear in either script
- **Character substitution evasion** — attackers replace characters (а → @, о → 0) to slip past keyword filters
- **No existing labeled dataset** — we had to build one from scratch

Smishi was built to fill this gap.

## How it works

Smishi uses an **ensemble** of two models plus rule-based heuristics:

| Component | Detail |
|---|---|
| Model A | TF-IDF (character n-grams, 3–5) + Logistic Regression |
| Model B | Fine-tuned BERTić transformer (`ravi2505/ne-nasedaj-sms-phishing`) |
| Heuristics | Suspicious/typosquatted domain detection, message-length analysis, threat-vector flags |
| Training data | 400+ labeled SBCM SMS messages (phishing + legitimate), Cyrillic & Latin |
| Languages | Serbian (Cyrillic & Latin), Bosnian, Croatian, Montenegrin |
| Infrastructure | Model A + heuristics run on CPU; Model B uses GPU if available, falls back to CPU |
| Interface | Bilingual (EN/SR) Gradio app, single-message and batch (CSV) modes |

Both model predictions and confidence scores are shown side by side, along with flagged red-flag indicators (suspicious domains, typosquatting, urgency language, etc.).

## Known limitations

- **No-URL phishing**: The system currently underperforms on scams that rely on IBAN manipulation, voice phishing (vishing) scripts, or social pressure without links. This is a known gap and active area of improvement.
- **Dataset size**: ~400 rows is a solid starting point but benefits from more diverse examples, especially Bosnian and Montenegrin regional variants.
- **Novel attack patterns**: AI-generated smishing may use phrasing outside the training distribution. Contributions welcome.

## Contributing

We welcome:
- New labeled SMS examples (phishing or legitimate) in any SBCM language
- Edge cases: messages without URLs, IBAN scams, impersonation without links
- Cyrillic-heavy examples, regional dialect variants

Open an issue or pull request on this repository.

## Built by

Utaem & Monkeydluffy
Built during the Build Small Hackathon, June 2026.

## References

- UNDP / PwC Serbia, *Tržište rada u oblasti sajber bezbednosti u Srbiji*, Belgrade, June 2026
- National CERT Serbia, Phishing campaign alert, August 2024
- M. Drolet, "AI Is Amping Up Phishing, Smishing and Vishing Attacks," Forbes Technology Council, May 2025
- RATEL, Annual Report 2024

---

# 🎣 Smishi — Ne Nasedaj (SR)

**Detektor SMS phishinga za srpski, bosanski, hrvatski i crnogorski**

*"Ne Nasedaj" — Don't Fall For It.*

## Šta je ovo?

Smishi je open-source detektor SMS phishinga (smishinga) napravljen specijalno za južnoslovenske jezike — srpski, bosanski, hrvatski i crnogorski.

Zalepi SMS poruku. Model ti kaže da li izgleda kao prevara, na srpskom ili engleskom (toggle u interfejsu).

Bez aplikacije. Bez prijave. Bez čuvanja podataka. Samo zalepi i provjeri.

## Zašto postoji?

Smishing je jedan od najbrže rastućih sajber pretnji na Zapadnom Balkanu. Kampanje koje imitiraju institucije poput Pošte Srbije ciljaju građane radi finansijske prevare i krađe kredencijala. Phishing napadi globalno su značajno porasli od 2022. godine, pojačani AI alatima koji lokalizovane, uverljive lažne poruke čine trivijalnim za masovnu produkciju.

Ipak, ne postoji javno dostupan alat za detekciju smishinga koji podržava naše jezike. Generički višejezični modeli imaju problema sa:

- **Bogatom morfološkom fleksijom** — *nagrada / nagradu / nagradi* su ista ključna reč napada u tri oblika
- **Ćirilica ↔ latinica dualnost** — ista poruka može biti napisana u oba pisma
- **Supstitucijom karaktera** — napadači zamenjuju slova (а → @, о → 0) da prevare filtere
- **Nepostojanjem označenog dataseta** — morali smo da ga izgradimo od nule

## Kako radi

Smishi koristi **ensemble** dva modela plus heuristike zasnovane na pravilima:

| Komponenta | Detalj |
|---|---|
| Model A | TF-IDF (karakterski n-grami, 3–5) + Logistička regresija |
| Model B | Fine-tuned BERTić transformer (`ravi2505/ne-nasedaj-sms-phishing`) |
| Heuristike | Detekcija sumnjivih/typosquat domena, analiza dužine poruke, threat-vector indikatori |
| Podaci | 400+ označenih SBCM SMS poruka (phishing + legitimne), ćirilica i latinica |
| Jezici | Srpski (ćirilica i latinica), bosanski, hrvatski, crnogorski |
| Infrastruktura | Model A + heuristike na CPU-u; Model B koristi GPU ako je dostupan, inače CPU |
| Interfejs | Dvojezični (EN/SR) Gradio app, pojedinačni i batch (CSV) režim |

## Poznata ograničenja

- **Phishing bez URL-a**: Sistem trenutno slabije prepoznaje prevare koje se oslanjaju na IBAN manipulaciju, vishing skripte ili socijalni pritisak bez linkova.
- **Veličina dataseta**: ~400 redova je dobra osnova, ali model profitira od više primera, posebno iz bosanskih i crnogorskih regionalnih varijanti.

## Doprinos

Dobrodošli su novi označeni SMS primeri, granični slučajevi (bez URL-a, IBAN prevare, imitacija bez linkova), ćirilični i regionalni dijalekatski primeri. Otvori issue ili pull request.

## Napravili

Utaem & Monkeydluffy
Napravljeno tokom Build Small Hackathona, juni 2026.

