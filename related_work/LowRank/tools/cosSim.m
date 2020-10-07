function [Sim] = cosSim(a, b)
Sim = (a*b') / (norm(a)*norm(b));
end