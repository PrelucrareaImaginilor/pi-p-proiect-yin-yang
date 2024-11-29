# Yin-Yang - Determinarea trasaturilor de personalitate prin analizarea scrisului de mana

## Analiza literaturii de specialitate:<br>

| Nr  | Autor(i)/An | Aplicatie/Domeniu | Tehnologii utilizate | Metodologie/Abordare | Rezultate | Limitari | Comentarii suplimentare |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | safvanck (2019)  | Handwriting Analysis for Detection of Personality Traits  | Keras Deep Learning  | Aplicatie Python  |  Extragere Trasaturi Scris  | Vechimea proiectului (5 ani)  | Proiect asemanator  |
| 2  | Tahar Mekhaznia, Chawki Djeddi, Sobhan Sarkar (2021)  | Personality Traits Identification Through Handwriting Analysis  | Artificial Neural Network   | Extragere Trasaturi Scris  | 5 trasaturi de personalitate  | -  | -  |
| 3  | Anna Esposito, Terry Amorese, Michele Buonanno, Marialucia Cuciniello, Antonietta M. Esposito, Gennaro Cordasco (2019)  | Handwriting and Drawing Features for Detecting Personality Traits  | Digitizing Tablet, Intuos Inkpen  | Analiza Semnale Digitale  | 5 trasaturi de personalitate  | -  | -  |
| 4  | Saniya Firdous (2022)  | Handwritten Character Recognition  | Convolutional Neural Network  | Segmentarea Imaginilor & Recunoașterea Optică | Analiza scrisului  | -  | Informatii Proiect Python  |
| 5  | Hilal	Müsevitoğlu, Ali	Öztürk, Fatiha	Nur	Başünald (2023)  | Detection	of	Personality	Features	From	Handwriting	By	Machine	Learning	Methods  | Artificial Neural Network  | Aplcatie Python  | Extragere Trasaturi Scris  | -  | Informatii Proiect Python  |

## Proiectarea soluției/aplicației

![Screenshot_81](https://github.com/user-attachments/assets/ab0ec66a-c6a4-4607-8e93-b25e205811a4)

Această schemă bloc are următoarele componente:

1. Imagine - Imaginea inițială cu scrisul de mână.
2. Preprocesare - Curăță imaginea de zgomote (linii de caiet, pete de cerneală) și o convertește în alb-negru.
3. Segmentare - Extrage cuvintele, identifică spațiile dintre cuvinte și litere, grosimea textului (presiunea aplicată pe foaie), și cursivitatea scrisului.
4. Raport privind personalitatea individului - Generarea unui raport pe baza trăsăturilor identificate în scris.

## Descrierea proiectului

Proiectul "Yin-Yang" se concentrează pe analiza scrisului de mână pentru a determina trăsături de personalitate. Inspirat din diverse lucrări și proiecte similare, folosește tehnici de preprocesare și segmentare a imaginilor pentru a extrage caracteristici relevante din scris. Obiectivul este de a crea o aplicație Python care să genereze un raport asupra personalității individului pe baza analizelor automate ale trăsăturilor de scris.

Link pentru seturile de date pentru antrenare si testare a proiectului in faza curenta: https://drive.google.com/file/d/1jriCei9Cg1jEigRgYw4VZGIrUJFigvT8/view?usp=sharing
