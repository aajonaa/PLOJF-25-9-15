%% Delete the legend if the experiment result had the legend (first step of processing)
% 2025-6-13

%% 打开Fig文件

% Define your array of indices

% Iterate over the array using a for loop
for i = 1:29
    index = i;
    % Your code that uses 'index' here
    fprintf('Current index: %d\n', index); % Example action using the index
    
    %% 设置打开的文件名
    openFilename = ['PLOJF-F', num2str(index), '-CC.fig'];
    % Open the .fig file and get a handle to the figure
    figHandle = openfig(openFilename, 'reuse');

    %% Delete legend (if it exists)
    legend_handle = findobj(gcf, 'Type', 'legend');
    if ~isempty(legend_handle)
        delete(legend_handle); % Remove legend completely
    end

    %% 保存Fig文件
    saveFilename = ['LegendOff-PLOJF-F', num2str(index), '-CC.fig'];
    % Saving the modified figure as a new .fig file using the handle
    savefig(figHandle, saveFilename);
    
    % Close the figure to free up resources
    close(figHandle);
end
