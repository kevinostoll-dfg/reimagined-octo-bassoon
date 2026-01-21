sec-api.io
SEC API by D2V
Products
Filings
Pricing
Sandbox
Docs
Tutorials
Log in
Get Free API Key
API Documentation
Introduction
Filing Query API
Full-Text Search API
Stream API
Download & PDF Generator API
XBRL-to-JSON Converter 
Extractor API 
Form ADV API - Investment Advisers
Form 3/4/5 API - Insider Trading
Overview
Endpoint & Authentication
└ Search API
└ Bulk Download
Request Parameters
└ Examples
Response Format
└ Examples
└ Bulk Downloads
└ Transaction Codes
Example: Python
Form 144 API - Restricted Sales
Form 13F API - Institut. Holdings
Form 13D/13G API - Activist Invst.
Form N-PORT API - Mutual Funds
Form N-CEN API - Annual Reports
Form N-PX API - Proxy Voting
Form S-1/424B4 API - IPOs, Notes
Form C API - Crowdfunding
Form D API - Private Sec. Offerings
Form 1-A/1-K/1-Z - Reg A Offerings
Form 8-K API - Item 4.01
Form 8-K API - Item 4.02
Form 8-K API - Item 5.02
Executive Compensation API
Directors & Board Members
Audit Reports (Fin. Stat. & ICFR)
Audit Fees API
Company Subsidiaries
Outstanding Shares & Public Float
SEC Enforcement Actions
SEC Litigation Releases
SEC Administrative Proceedings
AAER Database API
SRO Filings Database
CIK, CUSIP, Ticker Mapping API
EDGAR Entities Database
Financial Statements
Insider Trading Data from SEC Form 3, 4, 5 Filings
Insider Trading API
Form 4 XML converted to JSON
The Insider Trading Data API allows you to search and list all insider buy and sell transactions of all publicly listed companies on US stock exchanges. Insider activities of company directors, officers, 10% owners and other executives are fully searchable. The insider trading database includes information about the CIK and name of the insider, her/his relationship to the company, the number of shares and securities purchased or sold, the purchase or selling price, the date of the transaction, the amount of securities held before and after the transaction occured, any footnotes such as the effect of Rule 10b-18 or 10b5-1 stock purchase plans and more. The full list of all data points is available below.

Insider trades are reported to the SEC in Form 3, 4 and 5 filings. Our system converts the XML data in SEC Form 3, 4, and 5 into a standardized JSON format, indexes the information into our database and makes the data searchable through the Insider Trading Data API.

New insider transactions are added to the database in real-time as soon as the corresponding filing is published on SEC EDGAR.

Form 3
Form 4
Form 5
Example: Form 3 in JSON Format
{
  "accessionNo": "0000921895-24-002825",
  "filedAt": "2024-11-26T19:53:45-05:00",
  "documentType": "3",
  "periodOfReport": "2024-11-22",
  "notSubjectToSection16": false,
  "issuer": {
    "cik": "813298",
    "name": "DESTINATION XL GROUP, INC.",
    "tradingSymbol": "DXLG"
  },
  "reportingOwner": {
    "cik": "1959730",
    "name": "Fund 1 Investments, LLC",
    "address": { ... },
    "relationship": {
      "isDirector": false,
      "isOfficer": false,
      "isTenPercentOwner": true,
      "isOther": false
    }
  },
  "nonDerivativeTable": {
    "holdings": [
      {
        "securityTitle": "Common Stock, par value $0.01 per share",
        "postTransactionAmounts": {
          "sharesOwnedFollowingTransaction": 5720548
        },
        "ownershipNature": {
          "directOrIndirectOwnership": "I",
          "natureOfOwnership": "See Footnotes",
          "natureOfOwnershipFootnoteId": [ "F1", "F2" ]
        }
      }
      // ... more non derivative holdings
    ]
  },
  "derivativeTable": {
    "holdings": [
      {
        "securityTitle": "Cash-Settled Total Return Swap",
        "securityTitleFootnoteId": [ "F4" ],
        "conversionOrExercisePriceFootnoteId": [ "F4" ],
        "exerciseDateFootnoteId": [ "F4" ],
        "expirationDate": "2026-02-24",
        "expirationDateFootnoteId": [ "F5" ],
        "underlyingSecurity": {
          "title": "Common Stock, par value $0.01 per share",
          "shares": 5763573
        },
        "ownershipNature": {
          "directOrIndirectOwnership": "I",
          "natureOfOwnership": "See Footnotes",
          "natureOfOwnershipFootnoteId": [ "F1", "F2" ]
        }
      }
    ]
  },
  "footnotes": [
    {
      "id": "F1",
      "text": "Securities reported herein are held for the benefit of Pleasant Lake Onshore Feeder Fund, LP (the \"PL Fund\") and an additional private investment vehicle for which Pleasant Lake Partners LLC (\"PLP\") serves as investment adviser. Fund 1 Investments, LLC serves as managing member of PLP. Jonathan Lennon serves as managing member of Fund 1 Investments, LLC. Each of the Reporting Persons disclaims beneficial ownership of the securities reported herein except to the extent of its or his pecuniary interest therein."
    },
    {
      "id": "F2",
      "text": "Securities held for the account of the PL Fund."
    },
    {
      "id": "F3",
      "text": "Securities held for the account of an unaffiliated private fund for which PLP serves as investment adviser."
    },
    {
      "id": "F4",
      "text": "PL Fund has entered into certain cash-settled total return swap agreeements (the \"Swap Agreements\") with an unaffiliated third party financial institution, which provides PL Fund with economic exposure to an aggregate of 5,763,573 nominal shares of Common Stock. The Swap Agreements provide PL Fund with economic results that are comparable to the economic results of ownership but do not provide PL Fund with the power to vote or direct the voting or dispose of or direct the disposition of the shares of Common Stock that are the subject of the Swaps Agreements (the \"Subject Shares\"). The Reporting Persons expressly disclaim beneficial ownership of the Subject Shares except to the extent of its or his pecuniary interest therein."
    },
    {
      "id": "F5",
      "text": "The expiration date of the Swap Agreements will be automatically extended for successive 12 month periods unless one party provides written notice to the other party, at least 30 calendar days prior to the first extension and at least 15 calendar days prior to any subsequent extension, not to so extend the expiration date."
    }
  ]
}
The HTTP POST-based API accepts a search query and returns all matching insider transactions. All data points are searchable. For example, finding all insider trades by the ticker symbol of a company, looking for insider buying acticity in a particular industry or sector or monitoring transactions falling under Rule 10b5-1. The API returns a maximum of 50 insider transactions per query. Increase the from parameter by 50 each time you are requesting the next batch of matching transactions.

Dataset history:
All Form 3, 4 and 5 filings and amended versions filed since 2009 to present.
Data update frequency:
New insider transaction data is extracted, indexed, and made searchable within an average of 300 milliseconds after a new Form 3/4/5 filing is published.
Survivorship bias free:
Yes. The database includes all insider transactions, including from delisted companies.
API Endpoint
Search API
Retrieve insider trading data by sending a POST HTTP request with the search parameters as the payload to the following endpoint:

https://api.sec-api.io/insider-trading
Supported HTTP methods: POST

Request and response content type: JSON

Bulk Download APIs
The complete sets of Form 3, 4 and 5 filings are available for bulk download in compressed JSONL (JSON line) files (.jsonl.gz). Each line in a .jsonl.gz file represents the full content of a single Form 3, 4 or 5 filing in structured JSON format. The dataset is organized by year and month, using the filename format YYYY-MM.jsonl.gz, where YYYY is the year (e.g., 2024) and MM is the month (e.g., 02 for February). Each .jsonl.gz file includes all insider trading filings with a filedAt timestamp that falls within the respective year and month. This timestamp reflects when the filing was accepted by the EDGAR system.

New filings published during the previous day are added daily to the bulk datasets between 1:00 AM and 4:00 AM ET.

An accompanying index.json file provides metadata for all available .jsonl.gz files, including:

key (string) - The file path, e.g. 2025/2025-03.jsonl.gz.
updatedAt (date) - The last update timestamp, e.g. 2025-04-03T14:06:34.000Z.
size (integer) - The file size in bytes, e.g. 106764954.
The index.json file is especially useful for programmatic access and automation, allowing to monitor updates and manage downloads at scale.

Endpoint	Description	HTTP Method	Response Format
/bulk/form-3/YEAR/YEAR-MONTH.jsonl.gz	Gzip-compressed JSONL file containing all Form 3 data for the specified year and month.	GET	jsonl.gz
/bulk/form-3/index.json	JSON file containing the paths, file update times and file sizes of all jsonl.gz files of all Form 3 data files.	GET	json
/bulk/form-4/YEAR/YEAR-MONTH.jsonl.gz	Gzip-compressed JSONL file containing all Form 4 data for the specified year and month.	GET	jsonl.gz
/bulk/form-4/index.json	JSON file containing the paths, file update times and file sizes of all jsonl.gz files of all Form 4 data files.	GET	json
/bulk/form-5/YEAR/YEAR-MONTH.jsonl.gz	Gzip-compressed JSONL file containing all Form 5 data for the specified year and month.	GET	jsonl.gz
/bulk/form-5/index.json	JSON file containing the paths, file update times and file sizes of all jsonl.gz files of all Form 5 data files.	GET	json
Bulk Download Endpoint Examples
Open the following example URLs in your browser to download all Form 4 filings published in February 2025, or to download the index file containing all available Form 4 bulk dataset files. Replace YOUR_API_KEY with your actual API key before using the URLs.

https://api.sec-api.io/bulk/form-4/2025/2025-02.jsonl.gz?token=YOUR_API_KEY
https://api.sec-api.io/bulk/form-4/index.json?token=YOUR_API_KEY
Authentication
To authenticate API requests, use the API key displayed in your user profile. Utilize the API key in one of two ways. Choose the method that best fits your use case:

Authorization Header: Set the API key as an Authorization header. For instance, before sending a POST request to https://api.sec-api.io/insider-trading, ensure the header is set as follows: Authorization: YOUR_API_KEY. Do not include any prefix like Bearer.
Query Parameter: Alternatively, append your API key directly to the URL as a query parameter. For example, when making POST requests, use the URL https://api.sec-api.io/insider-trading?token=YOUR_API_KEY instead of the base endpoint.
Request Parameters
All insider transaction properties are searchable. Refer to the complete list of properties in the Response Structure section below. You can send a search query as JSON formatted payload to the API using the structure below.

Request parameters:

query (string) - The query string defines the search expression and is written in Lucene syntax. You can specify which insider transaction document fields to search by using the field name followed by a colon :. For example, to search for insider transactions disclosed by Tesla executives, you can use the following query: issuer.tradingSymbol:TSLA. The query language supports boolean operators (AND, OR, NOT), wildcards (*), range searches across date and number fields, and nested conditions. More information about the Lucene query syntax can be found in our tutorial here. Let's look at some examples:
The query reportingOwner.relationship.isDirector:True AND issuer.tradingSymbol:AMZN returns all insider transactions performed by any director at Amazon.
nonDerivativeTable.transactions.coding.code:A AND periodOfReport:[2021-01-01 TO 2021-06-30] returns all insider purchase transactions (transaction code "A") reported in the first six months of 2021.
from (integer) - Specifies the starting position in the results set for your query, functioning like an array index. Default is 0, with a maximum value of 10,000, which also represents the upper limit of results that can be returned per query. To paginate through results, increment the from value accordingly - for instance, setting from to 50 retrieves results from the 51st to the 100th transaction, assuming a size parameter of 50. If your total results exceed 10,000, consider refining your search criteria to reduce the result set. A common method is to use a date range filter on the periodOfReport field, such as periodOfReport:[2021-01-01 TO 2021-01-31], to paginate through all results for January 2021 by progressively increasing the from value (0, 50, 100, etc.). For subsequent months like February 2021, update your query to periodOfReport:[2021-02-01 TO 2021-02-28] and repeat the pagination process.
size (integer) - The number of transactions to be returned in one response. Default: 50. Max: 50.
sort (array) - Specifies the sorting order of the results. The default sort order is by the filedAt field in descending order, starting with the most recent: [{ "filedAt": { "order": "desc" } }]. The sort parameter is an array of sort definitions. Each array item defines the sort order for the result. Set the order property to asc for ascending order or desc for descending order. The field property specifies the field by which the results are sorted. For example, to sort the results by the periodOfReport field in ascending order, use the following sort definition: [{ "periodOfReport": { "order": "asc" } }].
Request Examples
Find the most recent insider trades reported by Tesla executives
The example search query "query": "issuer.tradingSymbol:TSLA" returns the most recent insider transactions reported by any officer, director, 10% owner and other insider working at Tesla. The API returns the first 50 matching transactions while the response size is limited to 50 transactions and sorted by the filedAt parameter.

JSON
{
    "query": "issuer.tradingSymbol:TSLA",
    "from": "0",
    "size": "50",
    "sort": [{ "filedAt": { "order": "desc" } }]
}
Response
Table
JSON
Company	Insider Name	Director	Officer	10% Owner	Transaction Codes	Acquired /
Disposed	$ Traded
(Non-Derivates)	Shares Transacted
(Non-Derivates)	Securities Traded
( Non-Derivative )	Trade Reported At	Trade Perfomed On	Has
Footnotes	...
TSLA	Taneja Vaibhav (Chief Financial Officer)		x		Exercise or Conversion, Sale (Open-Market)	A, D	$1,170,635.499	9,175	Common Stock	12/10/2025	12/5/2025	Y	...
TSLA	Musk Kimbal	x			Gift	D	$0	14,785	Common Stock	11/13/2025	11/10/2025	Y	...
TSLA	Musk Elon (CEO)	x	x	x	Grant	A	$141,568,600,887.36	423,743,904	Common Stock	11/11/2025	11/6/2025	Y	...
TSLA	MURDOCH JAMES R	x			Sale (Open-Market)	D	$25,360,800	60,000	Common Stock	9/18/2025	9/15/2025	Y	...
TSLA	Musk Elon (CEO)	x	x	x	Purchase (Open-Market)	A	$999,959,042.367	2,568,732	Common Stock	9/15/2025	9/12/2025	Y	...
TSLA	Zhu Xiaotong (SVP, APAC and)		x		Sale (Open-Market)	D	$7,275,100	20,000	Common Stock	9/13/2025	9/11/2025	Y	...
TSLA	Taneja Vaibhav (Chief Financial Officer)		x		Exercise or Conversion, Sale (Open-Market)	A, D	$918,136.512	9,143.5	Common Stock	9/10/2025	9/5/2025	Y	...
TSLA	MURDOCH JAMES R	x			Sale (Open-Market)	D	$42,034,320	120,000	Common Stock	8/29/2025	8/26/2025	Y	...
TSLA	Musk Elon (CEO)	x	x	x	Grant	A	$2,240,640,000	96,000,000	Common Stock	8/5/2025	8/3/2025	Y	...
TSLA	MURDOCH JAMES R	x			Exercise or Conversion	A, D	$1,572,300	90,000	Common Stock	7/16/2025	7/11/2025	Y	...
←
1
2
→
Find recent open-market insider purchases with code P
The following shows how to find all insider transactions representing open-market purchases of securities by insiders using transaction code "P". The example search query nonDerivativeTable.transactions.coding.code:P looks for all non-derivate transactions with transaction code "P", which stands for open market or private purchase of securities. Change the code parameter to "S" to find open-market insider sales. The full list of transaction codes is available below.

JSON
{
    "query": "nonDerivativeTable.transactions.coding.code:P",
    "from": "0",
    "size": "50",
    "sort": [{ "filedAt": { "order": "desc" } }]
}
Response
Table
JSON
Company	Insider Name	Director	Officer	10% Owner	Transaction Codes	Acquired /
Disposed	$ Traded
(Non-Derivates)	Shares Transacted
(Non-Derivates)	Securities Traded
( Non-Derivative )	Trade Reported At	Trade Perfomed On	Has
Footnotes	...
UUU	AULT MILTON C III	x			Purchase (Open-Market)	A	$130,081.77	29,100	Common Stock	12/11/2025	12/5/2025	Y	...
DOMH	Hayes Anthony (CEO)		x		Purchase (Open-Market)	A	$89,053.7	23,000	Common Stock	12/11/2025	12/8/2025	Y	...
DOMH	Wool Kyle Michael (President)		x		Purchase (Open-Market)	A	$96,797.5	25,000	Common Stock	12/11/2025	12/8/2025	Y	...
NYC	SCHORSCH NICHOLAS S			x	Purchase (Open-Market)	A	$25,930.8	3,240	Class A common stock	12/11/2025	12/8/2025	Y	...
AIRJ	Pang Stephen S. (Chief Financial Officer)		x		Purchase (Open-Market)	A	$6,398.55	2,169	Class A Common Stock	12/11/2025	12/10/2025	Y	...
AIRJ	JORE MATTHEW B (Chief Executive Officer)	x	x	x	Purchase (Open-Market)	A	$29,769.22	10,500	Class A Common Stock	12/11/2025	12/8/2025		...
CJMB	Duke Liberty Smith	x			Purchase (Open-Market), Grant	A	$2,334.24	1,512	Common Stock	12/11/2025	12/8/2025	Y	...
NEXT	Hanwha Aerospace Co., Ltd.			x	Purchase (Open-Market)	A	$5,742,413.653	932,598	Common Stock	12/11/2025	12/8/2025	Y	...
AIRJ	Porter Stuart D	x		x	Purchase (Open-Market)	A	$999,949.65	342,645	Class A Common Stock	12/11/2025	12/8/2025	Y	...
AMR	Courtis Kenneth S.	x			Purchase (Open-Market)	A	$6,309,389.87	36,000	Common Stock, $0.01 par value per share	12/11/2025	12/8/2025	Y	...
←
1
2
→
Find insider trades reported under Rule 10b5-1
The example search query "query": "footnotes.text:10b5-1" looks for all insider transactions with footnotes that include the term "10b5-1". The API returns the first 50 matching transactions while the response size is limited to 50 transactions and sorted by the filedAt parameter.

JSON
{
    "query": "footnotes.text:10b5-1",
    "from": "0",
    "size": "50",
    "sort": [{ "filedAt": { "order": "desc" } }]
}
Response
Table
JSON
Company	Insider Name	Director	Officer	10% Owner	Transaction Codes	Acquired /
Disposed	$ Traded
(Non-Derivates)	Shares Transacted
(Non-Derivates)	Securities Traded
( Non-Derivative )	Trade Reported At	Trade Perfomed On	Has
Footnotes	...
BCDA	Altman Peter (President and CEO)	x	x		Grant, F	A	$223,408.64	163,072	Common Stock	12/11/2025	12/8/2025	Y	...
BCDA	McClung David (Chief Financial Officer)		x		Grant, F	A	$120,032.55	87,615	Common Stock	12/11/2025	12/8/2025	Y	...
BCDA	GILLIS EDWARD M (Senior Vice President, Devices)		x		Grant, F	A	$72,952.5	53,250	Common Stock	12/11/2025	12/8/2025	Y	...
WVE	Rawcliffe Adrian	x			Exercise or Conversion, Sale (Open-Market)	A, D	$880,740	84,000	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	Wagner Heidi L	x			Exercise or Conversion, Sale (Open-Market)	A, D	$272,580	28,000	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	HENRY CHRISTIAN O	x			Sale (Open-Market), Exercise or Conversion	D, A	$1,810,113.75	180,445	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	Tan Aik Na	x			Sale (Open-Market), Exercise or Conversion	D, A	$2,401,428.6	253,448	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	Verdine Gregory L.	x			Sale (Open-Market)	D	$269,460.8	20,000	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	Francis Chris (See Remarks)		x		Exercise or Conversion, Sale (Open-Market)	A, D	$8,534,268.205	882,062	Ordinary Shares	12/11/2025	12/8/2025	Y	...
WVE	Moran Kyle (Chief Financial Officer)		x		Exercise or Conversion, Sale (Open-Market), Grant	A, D	$5,643,399	556,036	Ordinary Shares	12/11/2025	12/8/2025	Y	...
←
1
2
→
Response Structure
Response type: JSON

The transactions array includes all XML-to-JSON converted filings. Each array item represents a single SEC form 3, 4 or 5. Many data fields have optional footnotes attached. The "with footnote" mark indicates that the parameter has a footnote. For example, "securityTitle (string, with footnote)" indicates that securityTitle has footnotes. The corresponding footenote key is securityTitleFootnoteId. The ID and footnote content live in the array footnotes.

transactions (array) - An array of all matching transactions as reported in Form 3, 4 and 5. An array item represents the JSON converted XML data of a matching filing.
accessionNo (string) - Accession number of the original filing.
filedAt (dateTime) - The date and time when the transaction filing was accepted by SEC EDGAR. Example: 2022-08-09T21:23:00-04:00
documentType (string) - The type of the form: 3, 3/A, 4, 4/A, 5, 5/A.
periodOfReport (date) - Meaning in form 3: date of event requiring statement. In form 4: date of earliest transaction. In form 5: statement for issuer's fiscal year end. Format: YYYY-MM-DD
dateOfOriginalSubmission (date) - If amended, date of original filed of the format YYYY-MM-DD. Mandatory in form 3/A, 4/A and 5/A.
issuer (object) - Issuer information
cik (string) - CIK of issuer. Leading zeros are removed.
name (string) - Issuer name.
tradingSymbol (string) - Issuer trading symbol.
reportingOwner (object) - Information about the reporting entity.
cik (string) - CIK of reporting entity. Leading zeros are removed.
name (string) - Name of reporting entity.
relationship (object) - Relationship to issuer.
isDirector (boolean) - True if the reporting person is a director. False otherwise.
isOfficer (boolean) - True if the reporting person is an officer. False otherwise. If True, then officerTitle becomes mandatory.
officerTitle (string) - Officer title if isOfficer is True.
isTenPercentOwner (boolean) - True if the reporting person owns 10% of the issuer. False otherwise.
isOther (boolean) - True if the reporting person has a different relationship. False otherwise. If True, then the otherText property becomes mandatory.
otherText (string) - Explanation if isOther is True.
address (object) - Address of reporting entity.
street1 (string)
street2 (string)
ciy (string)
zipCode (string)
stateDescription (string)
nonDerivativeTable (object) - Table I - Non-Derivative Securities
transactions (array) - Non-derivative transactions
securityTitle (string, with footnote) - Title of security
transactionDate (date, with footnote) - Transaction date of the format YYYY-MM-DD.
deemedExecutionDate (date, with footnote) - Deemed execution date of the format YYYY-MM-DD.
coding (object) - Transaction code properties.
formType (string) - Valid values: 3, 3/A, 4, 4/A, 5, 5/A.
code (string) - See Code List for a list of possible values.
equitySwapInvolved (boolean)
footnoteId (string)
timeliness (string, with footnote) - Valid values: E = early, L = late, empty = on-time.
amounts (object)
shares (decimal, with footnote)
pricePerShare (decimal, with footnote)
acquiredDisposedCode (string, with footnote) - Valid values: A = acquired, D = disposed.
postTransactionAmounts (object) - Amount owned following reported transaction.
sharesOwnedFollowingTransaction (decimal, with footnote)
valueOwnedFollowingTransaction (decimal, with footnote)
ownershipNature (object)
directOrIndirectOwnership (string, with footnote) - Valid values: I = indirect, D = direct.
natureOfOwnership (string, with footnote)
holdings (array)
securityTitle (string, with footnote)
coding (object)
formType (string)
footnoteId (string)
postTransactionAmounts (object) - Amount owned following reported transaction.
sharesOwnedFollowingTransaction (decimal, with footnote)
valueOwnedFollowingTransaction (decimal, with footnote)
ownershipNature (object)
directOrIndirectOwnership (string, with footnote) - Valid values: I = indirect, D = direct.
natureOfOwnership (string, with footnote)
derivativeTable (object) - Table II - Derivative Securities
transactions (array)
securityTitle (string, with footnote)
conversionOrExercisePrice (decimal, with footnote)
transactionDate (date, with footnote)
deemedExecutionDate (date, with footnote)
coding (object) - Transaction code properties.
formType (string) - Valid values: 3, 3/A, 4, 4/A, 5, 5/A.
code (string) - See Code List for a list of possible values.
equitySwapInvolved (boolean)
footnoteId (string)
timeliness (string, with footnote) - Valid values: E = early, L = late, empty = on-time.
amounts (object)
shares (decimal, with footnote) - Securities acquired (A) or disposed of (D)
pricePerShare (decimal, with footnote) - Price of derivative security
acquiredDisposedCode (string, with footnote) - Valid values: A = acquired, D = disposed.
exerciseDate (date, with footnote) - Date exercisable and expiration date
expirationDate (date, with footnote)
underlyingSecurity (object)
title (string, with footnote)
shares (decimal, with footnote)
value (decimal, with footnote)
postTransactionAmounts (object) - Amount owned following reported transaction.
sharesOwnedFollowingTransaction (decimal, with footnote)
valueOwnedFollowingTransaction (decimal, with footnote)
ownershipNature (object)
directOrIndirectOwnership (string, with footnote) - Valid values: I = indirect, D = direct.
natureOfOwnership (string, with footnote)
holdings (array)
securityTitle (string, with footnote)
conversionOrExercisePrice (date, with footnote)
coding (object) - Transaction code properties.
formType (string) - Valid values: 3, 3/A, 4, 4/A, 5, 5/A.
code (string) - See Code List for a list of possible values.
equitySwapInvolved (boolean)
footnoteId (string)
exerciseDate (date, with footnote)
expirationDate (date, with footnote)
underlyingSecurity (object)
title (string, with footnote)
shares (decimal, with footnote)
value (decimal, with footnote)
postTransactionAmounts (object) - Amount owned following reported transaction.
sharesOwnedFollowingTransaction (decimal, with footnote)
valueOwnedFollowingTransaction (decimal, with footnote)
ownershipNature (object)
directOrIndirectOwnership (string, with footnote) - Valid values: I = indirect, D = direct.
natureOfOwnership (string, with footnote)
footnotes (array) - Array of footnote objects.
id (string) - ID
text (string) - Footnote
remarks (array) - Array of remark objects.
ownerSignatureName (string)
ownerSignatureNameDate (date)
total (object) - An object with two properties "value" and "relation". If "relation" equals "gte" (= greater than or equal), the "value" is always 10,000. It indicates that more than 10,000 filings match the query. In order to retrieve all filings, you have to iterate over the results using the "from" and "size" variables sent to the API. If "relation" equals "eq" (= equal), the "value" represents the exact number of filings matching the query. In this case, "value" is always less than 10,000. We don't calculate the exact number of matching filings for results greater than 10,000.
Response Examples
The example illustrates the response to the search query "query": "issuer.tradingSymbol:TSLA" and includes the most recent insider trades executed by Elon Musk at Tesla.

JSON
{
      "total": {
          "value": 489,
          "relation": "eq"
      },
      "transactions": [
          {
              "id": "026dc8fd804de46ef08b5bad594998c5",
              "accessionNo": "0000899243-22-028189",
              "filedAt": "2022-08-09T21:23:00-04:00",
              "schemaVersion": "X0306",
              "documentType": "4",
              "periodOfReport": "2022-08-09",
              "notSubjectToSection16": false,
              "issuer": {
                  "cik": "1318605",
                  "name": "Tesla, Inc.",
                  "tradingSymbol": "TSLA"
              },
              "reportingOwner": {
                  "cik": "1494730",
                  "name": "Musk Elon",
                  "address": {
                      "street1": "C/O TESLA, INC.",
                      "street2": "1 TESLA ROAD",
                      "city": "AUSTIN",
                      "state": "TX",
                      "zipCode": "78725"
                  },
                  "relationship": {
                      "isDirector": true,
                      "isOfficer": true,
                      "officerTitle": "CEO",
                      "isTenPercentOwner": true,
                      "isOther": false
                  }
              },
              "nonDerivativeTable": {
                  "transactions": [
                      {
                          "securityTitle": "Common Stock",
                          "transactionDate": "2022-08-09",
                          "coding": {
                              "formType": "4",
                              "code": "S",
                              "equitySwapInvolved": false
                          },
                          "amounts": {
                              "shares": 435,
                              "pricePerShare": 872.469,
                              "pricePerShareFootnoteId": [
                                  "F1"
                              ],
                              "acquiredDisposedCode": "D"
                          },
                          "postTransactionAmounts": {
                              "sharesOwnedFollowingTransaction": 155058484
                          },
                          "ownershipNature": {
                              "directOrIndirectOwnership": "I",
                              "natureOfOwnership": "by Trust",
                              "natureOfOwnershipFootnoteId": [
                                  "F2"
                              ]
                          }
                      },
                      {
                          "securityTitle": "Common Stock",
                          "transactionDate": "2022-08-09",
                          "coding": {
                              "formType": "4",
                              "code": "S",
                              "equitySwapInvolved": false
                          },
                          "amounts": {
                              "shares": 13292,
                              "pricePerShare": 874.286,
                              "pricePerShareFootnoteId": [
                                  "F3"
                              ],
                              "acquiredDisposedCode": "D"
                          },
                          "postTransactionAmounts": {
                              "sharesOwnedFollowingTransaction": 155045192
                          },
                          "ownershipNature": {
                              "directOrIndirectOwnership": "I",
                              "natureOfOwnership": "by Trust",
                              "natureOfOwnershipFootnoteId": [
                                  "F2"
                              ]
                          }
                      },
                      {
                          "securityTitle": "Common Stock",
                          "transactionDate": "2022-08-09",
                          "coding": {
                              "formType": "4",
                              "code": "S",
                              "equitySwapInvolved": false
                          },
                          "amounts": {
                              "shares": 6048,
                              "pricePerShare": 876.629,
                              "pricePerShareFootnoteId": [
                                  "F4"
                              ],
                              "acquiredDisposedCode": "D"
                          },
                          "postTransactionAmounts": {
                              "sharesOwnedFollowingTransaction": 155039144
                          },
                          "ownershipNature": {
                              "directOrIndirectOwnership": "I",
                              "natureOfOwnership": "by Trust",
                              "natureOfOwnershipFootnoteId": [
                                  "F2"
                              ]
                          }
                      }
                  ]
              },
              "footnotes": [
                  {
                      "id": "F1",
                      "text": "The price reported in Column 4 is a weighted average price. These shares were sold in multiple transactions at prices ranging from $872.210 to $872.610, inclusive. The reporting person undertakes to provide Tesla, Inc., any security holder of Tesla, Inc. or the staff of the Securities and Exchange Commission, upon request, full information regarding the number of shares sold at each separate price within the range set forth in this footnote."
                  },
                  {
                      "id": "F2",
                      "text": "The Elon Musk Revocable Trust dated July 22, 2003, for which the reporting person is trustee."
                  },
                  {
                      "id": "F3",
                      "text": "The price reported in Column 4 is a weighted average price. These shares were sold in multiple transactions at prices ranging from $873.660 to $874.640, inclusive. The reporting person undertakes to provide Tesla, Inc., any security holder of Tesla, Inc. or the staff of the Securities and Exchange Commission, upon request, full information regarding the number of shares sold at each separate price within the range set forth in this footnote."
                  },
                  {
                      "id": "F4",
                      "text": "The price reported in Column 4 is a weighted average price. These shares were sold in multiple transactions at prices ranging from $876.100 to $876.925, inclusive. The reporting person undertakes to provide Tesla, Inc., any security holder of Tesla, Inc. or the staff of the Securities and Exchange Commission, upon request, full information regarding the number of shares sold at each separate price within the range set forth in this footnote."
                  }
              ],
              "remarks": "This Form 4 is the second of two Form 4s being filed by the Reporting Person relating to the same event. The Form 4 has been split into two filings to cover all 33 individual transactions that occurred on the same Transaction Date, because the SEC's EDGAR filing system limits a single Form 4 to a maximum of 30 separate transactions. Each Form 4 will be filed by the Reporting Person.",
              "ownerSignatureName": "By: Aaron Beckman by Power of Attorney For: Elon Musk",
              "ownerSignatureNameDate": "2022-08-09"
          }
      ]
  }
Response Structure of Bulk Download Endpoints
/bulk/form-3/YEAR/YEAR-MONTH.jsonl.gz
Decompressed JSONL Example of Form 3 Filings
{"id":"764b5b87fa93363a96475ce55c9754c7","accessionNo":"0000950170-25-048019","filedAt":"2025-03-31T21:30:04-04:00","schemaVersion":"X0206","documentType":"3","periodOfReport":"2025-03-20","notSubjectToSection16":false,"issuer":{"cik":"65270","name":"METHODE ELECTRONICS INC","tradingSymbol":"MEI"},"reportingOwner":{"cik":"2063597","name":"Erwin John Thomas","address":{"street1":"8750 W. BRYN MAWR AVE.","street2":"SUITE 1000","city":"CHICAGO","state":"IL","zipCode":"60631"},"relationship":{"isDirector":false,"isOfficer":true,"officerTitle":"CPO & EHS Officer","isTenPercentOwner":false,"isOther":false}},"nonDerivativeTable":{"holdings":[{"securityTitle":"Common Stock","coding":{},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":33156,"sharesOwnedFollowingTransactionFootnoteId":["F1"]},"ownershipNature":{"directOrIndirectOwnership":"D"}},{"securityTitle":"Common Stock","coding":{},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":1638},"ownershipNature":{"directOrIndirectOwnership":"I","natureOfOwnership":"Held in Methode 401(k) Plan"}}]},"footnotes":[{"id":"F1","text":"Represents Restricted Stock Units ("RSUs") which were previously granted on September 11, 2024 and January 15, 2025. The RSUs granted on September 11, 2024 totaled 32,200, of which 14,400 will vest on September 11, 2025, 14,400 will vest on September 11, 2026 and 3,400 will vest on September 11, 2027. The RSUs granted on January 15, 2025 totaled 956 and will vest ratably over 3 years from the grant date."}],"ownerSignatureName":"/s/ John Thomas Erwin","ownerSignatureNameDate":"2025-03-31"}
{ ... }
{ ... }
/bulk/form-3/index.json
index.json Example
[
    {
        "key": "2025/2025-04.jsonl.gz",
        "updatedAt": "2025-04-09T05:00:06.000Z",
        "size": 74929
    },
    {
        "key": "2025/2025-03.jsonl.gz",
        "updatedAt": "2025-04-08T11:00:53.000Z",
        "size": 227586
    }
	// ... more files
]
/bulk/form-4/YEAR/YEAR-MONTH.jsonl.gz
Decompressed JSONL Example of Form 4 Filings
{"id":"83d6d1d7e910e293aa3cd2a62bd5a46d","accessionNo":"0001104659-25-030101","filedAt":"2025-03-31T21:54:47-04:00","schemaVersion":"X0508","documentType":"4","periodOfReport":"2025-03-31","notSubjectToSection16":false,"issuer":{"cik":"1840502","name":"Taboola.com Ltd.","tradingSymbol":"TBLA"},"reportingOwner":{"cik":"1449433","name":"Apollo Management Holdings GP, LLC","address":{"street1":"9 WEST 57TH STREET, 43RD FLOOR","city":"NEW YORK","state":"NY","zipCode":"10019"},"relationship":{"isDirector":false,"isOfficer":false,"isTenPercentOwner":true,"isOther":false}},"nonDerivativeTable":{"transactions":[{"securityTitle":"Non-Voting Ordinary Shares, No Par Value","transactionDate":"2025-03-31","coding":{"formType":"4","code":"J","equitySwapInvolved":false,"footnoteId":["F1"]},"amounts":{"shares":1185242,"pricePerShare":3.01,"acquiredDisposedCode":"D"},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":40054344},"ownershipNature":{"directOrIndirectOwnership":"I","natureOfOwnership":"See Footnote","natureOfOwnershipFootnoteId":["F2"]}}],"holdings":[{"securityTitle":"Ordinary Shares, No Par Value","coding":{},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":39525691},"ownershipNature":{"directOrIndirectOwnership":"I","natureOfOwnership":"See Footnote","natureOfOwnershipFootnoteId":["F2"]}}]},"footnotes":[{"id":"F1","text":"The reported sales are between the Issuer and College Top Holdings, Inc., as part of the Issuer's share repurchase program and are intended to keep the Reporting Persons' ownership of Taboola's outstanding shares from reaching 25% or more. See Exhibit 99.1 for more information."},{"id":"F2","text":"See Exhibit 99.1."}],"ownerSignatureName":"see signatures attached as Exhibit 99.2","ownerSignatureNameDate":"2025-03-31"}
{ ... }
{ ... }
/bulk/form-4/index.json
index.json Example
[
    {
        "key": "2025/2025-04.jsonl.gz",
        "updatedAt": "2025-04-09T05:00:08.000Z",
        "size": 1675134
    },
    {
        "key": "2025/2025-03.jsonl.gz",
        "updatedAt": "2025-04-08T10:59:20.000Z",
        "size": 6142697
    }
	// ... more files
]
/bulk/form-5/YEAR/YEAR-MONTH.jsonl.gz
Decompressed JSONL Example of Form 5 Filings
{"id":"06d31397a5aa533faf912fe6ee249c80","accessionNo":"0001641172-25-001401","filedAt":"2025-03-28T20:23:39-04:00","schemaVersion":"X0508","documentType":"5","periodOfReport":"2025-02-28","notSubjectToSection16":false,"issuer":{"cik":"788611","name":"NextTrip, Inc.","tradingSymbol":"NTRP"},"reportingOwner":{"cik":"1563607","name":"Monaco Donald P","address":{"street1":"3900 PASEO DEL SOL","city":"SANTA FE","state":"NM","zipCode":"87507"},"relationship":{"isDirector":true,"isOfficer":false,"isTenPercentOwner":false,"isOther":false}},"derivativeTable":{"transactions":[{"securityTitle":"Series L Nonvoting Convertible Preferred Stock","conversionOrExercisePriceFootnoteId":["F1"],"transactionDate":"2024-12-31","coding":{"formType":"4","code":"A","equitySwapInvolved":false,"footnoteId":["F2"]},"timeliness":"L","exerciseDateFootnoteId":["F1"],"expirationDateFootnoteId":["F3"],"underlyingSecurity":{"title":"Common Stock","shares":413907},"amounts":{"shares":413907,"pricePerShare":3.02,"pricePerShareFootnoteId":["F2"],"acquiredDisposedCode":"A"},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":745032},"ownershipNature":{"directOrIndirectOwnership":"I","natureOfOwnership":"By Donald P. Monaco Insurance Trust","natureOfOwnershipFootnoteId":["F4"]}},{"securityTitle":"Series L Nonvoting Convertible Preferred Stock","conversionOrExercisePriceFootnoteId":["F1"],"transactionDate":"2025-02-24","coding":{"formType":"4","code":"A","equitySwapInvolved":false,"footnoteId":["F5"]},"timeliness":"L","exerciseDateFootnoteId":["F1"],"expirationDateFootnoteId":["F3"],"underlyingSecurity":{"title":"Common Stock","shares":331125},"amounts":{"shares":331125,"pricePerShare":3.02,"pricePerShareFootnoteId":["F5"],"acquiredDisposedCode":"A"},"postTransactionAmounts":{"sharesOwnedFollowingTransaction":745032},"ownershipNature":{"directOrIndirectOwnership":"I","natureOfOwnership":"By Donald P. Monaco Insurance Trust","natureOfOwnershipFootnoteId":["F4"]}}]},"footnotes":[{"id":"F1","text":"Shares of Series L Nonvoting Convertible Preferred Stock (Series L Preferred) shall not be convertible into shares of common stock unless and until stockholder approval of the conversion of the Series L Preferred into common stock ("Stockholder Approval") is obtained. Following receipt of Stockholder Approval, each share of Series L Preferred will automatically convert into one share of common stock, subject to certain limitations."},{"id":"F2","text":"On December 31, 2024, the Issuer and the Reporting Person entered into a debt conversion agreement, pursuant to which $1.25 million in existing promissory notes owed to the Reporting Person for monies advanced to the Issuer were converted into 413,907 shares of Series L Preferred at a price of $3.02 per share. The debt conversion agreement and the conversion of the promissory notes into shares of Series L Preferred were approved in advance by the Issuer's board of directors."},{"id":"F3","text":"The shares of Series L Preferred do not expire."},{"id":"F4","text":"The shares of Series L Preferred are beneficially owned by the Donald P. Monaco Insurance Trust (the "Trust"). Mr. Monaco, is the trustee of the Trust. As such, Mr. Monaco is deemed to beneficially own the shares held by the Trust. Mr. Monaco disclaims beneficial ownership of all securities held the Trust in excess of his pecuniary interest, if any, and this report shall not be deemed an admission that he is the beneficial owner of, or has pecuniary interest in, any such excess shares for the purposes of Section 16 of the Securities Exchange Act of 1934, as amended, or for any other purpose."},{"id":"F5","text":"On February 24, 2025, the Issuer and the Reporting Person entered into a debt conversion agreement, pursuant to which $1.00 million in existing promissory notes owed to the Reporting Person for monies advanced to the Issuer were converted into 331,125 shares of Series L Preferred at a price of $3.02 per share. The debt conversion agreement and the conversion of the promissory notes into shares of Series L Preferred were approved in advance by the Issuer's board of directors."}],"ownerSignatureName":"/s/ Donald P. Monaco","ownerSignatureNameDate":"2025-03-28"}
{ ... }
{ ... }
/bulk/form-5/index.json
index.json Example
[
    {
        "key": "2025/2025-04.jsonl.gz",
        "updatedAt": "2025-04-09T05:00:06.000Z",
        "size": 4456349
    },
    {
        "key": "2025/2025-03.jsonl.gz",
        "updatedAt": "2025-04-01T05:00:07.000Z",
        "size": 106764954
    },
    // ... more files
]
Transaction Code List
Value	Meaning
A	Grant, award or other acquisition pursuant to Rule 16b-3(d)
C	Conversion of derivative security
D	Disposition to the issuer of issuer equity securities pursuant to Rule 16b-3(e)
E	Expiration of short derivative position
F	Payment of exercise price or tax liability by delivering or withholding securities incident to the receipt, exercise or vesting of a security issued in accordance with Rule 16b-3
G	Bona fide gift
H	Expiration (or cancellation) of long derivative position with value received
I	Discretionary transaction in accordance with Rule 16b-3(f) resulting in acquisition or disposition of issuer securities
J	Other acquisition or disposition (describe transaction)
L	Small acquisition under Rule 16a-6
M	Exercise or conversion of derivative security exempted pursuant to Rule 16b-3
O	Exercise of out-of-the-money derivative security
P	Open market or private purchase of non-derivative or derivative security
S	Open market or private sale of non-derivative or derivative security
U	Disposition pursuant to a tender of shares in a change of control transaction
W	Acquisition or disposition by will or the laws of descent and distribution
X	Exercise of in-the-money or at-the-money derivative security
Z	Deposit into or withdrawal from voting trust

References
For more information about Form 3, 4 and 5 visit the SEC websites here:

Form 3 EDGAR PDF Template
Form 4 EDGAR PDF Template
Form 5 EDGAR PDF Template
Form 4 Filing Instructions
Form 5 Filing Instructions
Form 3, 4 and 5 Overview
Form 3, 4 and 5 XML Technical Specifications
Regulations
§ 240.10b5-1 Trading “on the basis of” material nonpublic information in insider trading cases
§ 240.16a-3 Reporting transactions and holdings
Research Papers
SEC Rule 10b5-1 and insiders' strategic trade
Offensive Disclosure: How Voluntary Disclosure Can Increase Returns from Insider Trading
Decoding Inside Information
Do SEC's 10b5-1 Safe Harbor Rules Need To Be Rewritten?
SEC Rule 10b5-1 Plans and Strategic Trade around Earnings Announcements
Insider sales based on short-term earnings information
Footer
Products
EDGAR Filing Search API
Full-Text Search API
Real-Time Filing Stream API
Filing Download & PDF Generator API
XBRL-to-JSON Converter
10-K/10-Q/8-K Item Extractor
Investment Adviser & Form ADV API
Insider Trading Data - Form 3, 4, 5
Restricted Sales Notifications - Form 144
Institutional Holdings - Form 13F
Form N-PORT API - Investment Company Holdings
Form N-CEN API - Annual Reports by Investment Companies
Form N-PX API - Proxy Voting Records
Form 13D/13G API
Form S-1/424B4 - IPOs, Debt & Rights Offerings
Form C - Crowdfunding Offerings
Form D - Private Placements & Exempt Offerings
Regulation A Offering Statements API
Changes in Auditors & Accountants
Non-Reliance on Prior Financial Statements
Executive Compensation Data API
Audit Fees Data API
Directors & Board Members Data
Company Subsidiaries Database
Outstanding Shares & Public Float
SEC Enforcement Actions
Accounting & Auditing Enforcement Releases (AAERs)
SRO Filings
CIK, CUSIP, Ticker Mapping
General
Pricing
Features
Supported Filings
EDGAR Filing Statistics
Account
Sign Up - Start Free Trial
Log In
Forgot Password
Developers
API Sandbox
Documentation
Resources & Tutorials
Python API SDK
Node.js API SDK
Legal
Terms of Service
Privacy Policy

SEC API

© 2025 sec-api.io by Data2Value GmbH. All rights reserved.

SEC® and EDGAR® are registered trademarks of the U.S. Securities and Exchange Commission (SEC).

EDGAR is the Electronic Data Gathering, Analysis, and Retrieval system operated by the SEC.

sec-api.io and Data2Value GmbH are independent of, and not affiliated with, sponsored by, or endorsed by the U.S. Securities and Exchange Commission.

sec-api.io is classified under SIC code 7375 (Information Retrieval Services), providing on-demand access to structured data and online information services.

