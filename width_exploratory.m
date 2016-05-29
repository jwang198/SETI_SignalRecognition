t = 5; %width in each direction

files = dir('/Users/Jason/Desktop/SETI_TimeSeries/SquiggleExamples')
for i = 1:size(files)
    file = files(i)
    disp(file.name)
    if (size(file.name) < 20)
        continue;
    end
    A = double(imread(file.name));
    
    for i = 50:113
        [max_intensity,max_index] = max(A(i,:));
        %subplot(16,8,i)
        subplot(8,8,i-49);
        plot(A(i, max(max_index-20,1):min(max_index+20,768)));
    end
end
