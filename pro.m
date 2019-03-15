%PROJEKAT IZ PREPOZNAVANJA OBLIKA
    % Predikcija pola
% Ivana Stojanovic EE 59/2014 
% Ljiljana Popovic EE 72/2014

%ucitavanje podataka
    X = load('mdc-gender-selected-features.mat'); % ucitavanje podataka
    % X je struktura koja ima data (78x41) i features u kome se nalaze nazivi
    % obelezja
    podaci = X.data(:,1:end-1);  
    labele=logical((X.data(:,end))-1);
    lab=(X.data(:,end));
    dummy= dummyvar(lab);
    obelezja=X.features; % nazivi obelezja
    z= zscore(podaci); % z-normalizovani podaci
    zene= z(labele(:,1)==0,:); 
    muskarci= z(labele(:,1)==1,:);

rng(1);
c = cvpartition(labele,'HoldOut',0.15);
Te = c.test; % labele koje se koriste kao test
Tr = c.training; % labele koje koristimo kao trening

Test = podaci(Te==1,:); % podaci za test izdvojeni pomocu logickog indeksiranja
Trening = podaci(Tr==1,:); %podaci za trening
TestLabele = labele(Te==1); %test labele
TreningLabele = labele(Tr==1); %trening labele
    
% % Prikaz raspodela estimiran Parzenovim prozorom
% h=0.7; % sirina prozora
% korak=0.1;
% for i = 1 : 40
%     [procena, min, max] = estimacija_1D_kde(zene(:,i),h, korak);
%     figure, plot(min-h/2:korak: max+h/2, procena,'r'), hold on
%     [procena, min, max] = estimacija_1D_kde(muskarci(:,i),h, korak);
%     plot(min-h/2:korak: max+h/2, procena,'b')
%     title (['Raspodela obelezja: ' X.features(i)])
%     legend('Zene','Muskarci')
% end 


%KNN klasifikatori
knn(Test,Trening,TestLabele, TreningLabele,25);
knn_l(Test,Trening,TestLabele, TreningLabele,25);
%SVM klasifikatori
svm(Test,Trening,TestLabele, TreningLabele);
svm_l(Test,Trening,TestLabele, TreningLabele);
% Random forest
random(Test,Trening,TestLabele,TreningLabele,80);
% Neuralne mreze

