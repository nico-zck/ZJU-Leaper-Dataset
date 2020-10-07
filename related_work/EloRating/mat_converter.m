clear;

mats = dir('*.mat');
for i=1:length(mats)
    clear names results;
    load(mats(i).name)
    save(mats(i).name, 'results', 'names', '-v7.3');
end
