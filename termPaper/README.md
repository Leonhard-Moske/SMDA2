# advanced methods of data analysis
## Term paper 4: Normalizing Flows

### TODO

* Tex anlegen mit Template
* literatur raussuchen
    1. astro theorie -> maybe ruth?
    2. normalizing flows für density estimation
    3. paper aus der Anleitung lesen: https://github.com/LukasRinder/normalizing-flows
* kurze Theory dataset
* dataset ansehen
    1. plots

### Tex

#### Abstract

"motivation why this method is helpfull"

#### Theory
1. normalizing flows
    Man wird wohl aus einem Prior ziehen und dann den normalizing flow darauf anwenden. Dabei die dkl verwenden als loss function. Ich weiß noch nicht
    was die ziel distribution seien soll. Ne das Ziel ist es ja einen classifier zu bauen. Dabei wollen wir von einem satz aus features zu einer pdf kommen.
	
	15.06: Neuer Stand: wir haben einen gaussian prior und wir haben die transformierte vairable. Wir trainieren unseren Flow so, dass die Data die transformierte 	Variable ist. Dann kann man die log likelyhood bei neuen daten auswerten (quasi beim prior) und diese ist dann eine gute wahrscheinlichkeitsverteilung. Dabei haben wir keine density distr. angenommen außer die Data. Wir mappen quasi unsere Data zu dem prior welcher dann eine gute probability distr ist. Dann kann man das bei allen classes machen und dann teilen 
2. astro stars

#### What I did

#### Findings

#### Refrences !!

* https://github.com/LukasRinder/normalizing-flows
* https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf  
* https://deepgenerativemodels.github.io/notes/flow/

### code

führe ich weiter aus wenn ich die theorie gelesen habe.
-> objektstruktur


### Protokoll / Notizen

* 12.06: Start mit aussuchen. README erstellt. Als erstes wohl normalizing flow theorie lesen.
* 16.06: habe den algo jetzt verstanden. Werde am Freitag nachfragen. Als nächstes wohl tensorflow angucken -> tutorial.
* 23.06: Habe mir das Tutorial angeguckt. schein nicht sooo komplex. Ich denke sobald man die Struktur von den bijektors verstanden hat ist es nicht so schwer. Wahrscheinlich am besten bei dem tutorial anfangen und dann über train utils runter gehen. Will als haupt Füller des Papers verschiedene Flows vergleichen.
	Tex template von computational physics übernommen, Abstact angefangen und sections beschriftet.	
	Will das mit der TODO-Methode von Ben machen.
	Todos eingefügt. Nicht alle die ich machen muss.
* 24.06: Habe eine website für die verschiedenen Flows gefunden. Will am Ende rumpspielen mit random freezing und systematic freezing usw. Fasse die Arten die ich gefunden habe in Onenote zusammen.

### bullshit additional ideas
