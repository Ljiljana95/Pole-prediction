function [p_procena, x_min, x_max] = estimacija_1D_kde(X, h, korak_raspodele)
%
% Funkcija koja procenjuje gustinu raspodele verovatnoce pomocu Parzenovog 
% prozora sirine h u slucaju nepoznate 1D raspodele koja je generisala 
% uzorke X.

x_min = min(X);
x_max = max(X);
N = size(X,1);

d = 1;  % 1D prostor obelezja
k = 1;

% Zauzimanje memorije za procenu gustine verovatnoce
p_procena = zeros(size(x_min-h/2:korak_raspodele:x_max+h/2));


for x=x_min-h/2:korak_raspodele:x_max+h/2
    
    unutar_prozora = abs(repmat(x, size(X)) - X) < h/2;
    
    % Formula za procenu gustine verovatnoce
    p_procena(k) = sum(unutar_prozora) * (1/N) * (1/h^d);
    
    k = k + 1;    
end

end