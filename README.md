# How Polysemous a Subword Is?

## Overview
This programms are for calculation of how polysemous a language is based of subword tokens.


## Directories
This program is processed following the four steps below:
1. Picks up a random Wikipedia article,
2. Tokenizes them into subword through an algorithm called WordPiece, and gets embeddings for each subword by BERT,
3. reduces their dimensions by tSNE, and estimates the number of meanings for each subword, and 
4. calculates the Shannon entropy of each subword based on the frequencies of how often each subword is used as a certain meaning.

## Datasets
The datasets this program uses are ranodom Wikipedia artibles.
The 88 languages below are available, which are assigned with ISO code in the program:

| Language | ISO code | Language | ISO code |
|-----|-----|-----|-----|
| Afrikaans | af | Kazakh | kk |
| Albanian | sq | Kirghiz | ky |
| Arabic | ar | Korean | ko |
| Aragonese | an | Latin | la |
| Armenian | hy | Latvian | lv |
| Azerbaijani | az | Lithuanian | lt |
| Bashkir | ba | Luxembourgish | lb |
| Basque | eu | Macedonian | mk |
| Belarusian | be | Malagasy | mg |
| Bengali | bn | Malay | ms |
| Bosnian | bs | Malayalam | ml |
| Breton | br | Marathi | mr |
| Bulgarian | bg | Nepali | ne |
| Burmese | my | Norwegian (Bokmal) | nb |
| Catalan | ca | Norwegian (Nynorsk) | nn |
| Chechen | ce | Occitan | oc |
| Chinese (Simplified) | zh | Persian (Farsi) | fa |
| Chinese (Traditional) | zh | Polish | pl |
| Chuvash | cv | Portuguese | pt |
| Croatian | hr | Punjabi | pa |
| Czech | cs | Romanian | ro |
| Danish | da | Russian | ru |
| Dutch | nl | Serbian | sr |
| English | en | Serbo-Croatian | sh |
| Estonian | et | Slovak | sk |
| Finnish | fi | Slovenian | sl |
| French | fr | Spanish | es |
| Galician | gl | Sundanese | su |
| Georgian | ka | Swahili | sw |
| German | de | Swedish | sv |
| Greek | el | Tagalog | tl |
| Gujarati | gu | Tajik | tg |
| Haitian | ht | Tamil | ta |
| Hebrew | he | Tatar | tt |
| Hindi | hi | Telugu | te |
| Hungarian | hu | Turkish | tr |
| Icelandic | is | Ukrainian | uk |
| Ido | io | Urdu | ur |
| Indonesian | id | Uzbek | uz |
| Irish | ga | Vietnamese | vi |
| Italian | it | Volapük | vo |
| Japanese | ja | Welsh | cy |
| Javanese | jv | West Frisian | fy |
| Kannada | kn | Yoruba | yo |

## How to Run
The python code can be run in the command below:

```
python wiki-clst.py 'path' 100 'id'
```
, in which `path` refers to a path of the .csv file containing language names, and `id` to the name of directory that the result will be output.

There are also optional arguments.

| Argument | Function |
|-----|-----|
| `--gpu` | avoiding using GPU |
| `--min_emb` | minimum embedings that will be processed |
| `--min_samples` | minimum datapoints in DBSCAN clustering |
| `--eps` | epsilon of DBSCAN clustering |
| `--dif` | the differenciation of the epsilons |
| `--tsne` | avoiding tSNE dimensional reduction |
| `--p_ratio` | the ratio to the number of embeddings defining perplexity in tSNE |
| `--save_tsne` | avoiding saving tSNE results |
| `-save_cluster` | saving the clustering result |

## Citation
```
@conference{nakayama-2025-jass,
	author          = {Nakayama, Takuto},
	booktitle       = {The 49th Annual of Meeting the Japanese Association of Sociolinguistic Sciences},
	title           = {言語は等しく多義的か？
	―サブワードと分散意味論に基づく形式–意味対応の分析― [Are languages equaly polysemous?: An analysis of form-meaning pairings based on subword tokens and distributional semantics]},
	year            = {2025},
	month           = {March},
	address         = {Tokyo, Japan},
	note			= {oral presentation}
}

@conference{nakayama-2025-nlp,
	author          = {Nakayama, Takuto},
	booktitle       = {The 31th Annual Meeting of the Association of Natural Language Processing},
	title           = {言語一般の計測を目指して: サブワードと分散意味論に基づく言語の複雑性計測 [Toward a linguistically general measurement: Measurement of linguistic complexity based on subword tokens and distributional semtantics]},
	year            = {2025},
	month           = {March},
	address         = {Nagasaki, Japan},
	note			= {poster presentation}
}
```