# SIIMHackathon2021

## Elevator pitch:
Flask based web app to query PACS via DICOMweb. 
Trachea + endotracheal tube segmentation real time.*

* on the way...


## Rationale
The clinical networks are usually closed ecosystems and the clinicians have only resticted user rights on the workstations. 
This environment does not allow to easily access the PACS to search and retrieve DICOM data programatically.

We found a way, that utilizes WinPthon, a portable, full blown python environment on Windows operating system.
You can ~~install~~ <-- no install, just unzip ;) and use it from here: https://winpython.github.io/

In the console simply install the dependencies and you can start making queries to the SIIM Hackathlon server to retrieve DICOM data via webDICOM protocoll.

In order to access the SIIM Hackathlon severs, you have to register on thei website: https://siim.org/general/custom.asp?page=hacking_healthcare


##TODO

- [x] query for image data on SIIM Hackathon server via webDICOM protocols
- [x] implement CORS (Cross-origin resource sharing) compliant way to do above
- [x] implement client side logic in Javascript for populating the site by iterating through the json WADO response
- [x] implement sequential (!) promise processing instead of parallel ( --> resource sparing for Flask server, memory problems(?) when loading model paralelly)
- [x] GUI goodies with color coding different results (anomaly vs. no anomaly vs. nothing detected classes)
- [x] add "privacy mode" by adding class with "blur" style to sensitive information & image data
- [ ] ad semantic segmentation library into pipeline
- [x] add HTML5 dektop notificatio by anomaly detection
- [ ] add hoplink to case by clicking "show case" button (alternatively run CLI command to open dicom viewer to review case)
- [ ] ... ?


 ![showcase](https://github.com/sandorkonya/SIIMHackathon2021/blob/31acfa10fb2cae9560d216dcc6790f8e59412b1e/showreel.gif)
