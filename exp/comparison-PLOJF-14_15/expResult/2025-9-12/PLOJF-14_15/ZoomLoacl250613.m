%% Zoom local range after the legend is off and grid if off (The second step of processing)
% 2025-6-13

%% 创建一个副本在Fig中间并删掉所有tick + Arrow

% Iterate over the array using a for loop
for i = 1:29
    index = i;
    % Your code that uses 'index' here
    fprintf('Current index: %d\n', index); % Example action using the index
    
    %% 设置打开的文件名
    openFilename = ['LegendOff-PLOJF-F', num2str(index), '-CC.fig'];
    figHandle = openfig(openFilename, 'reuse');

    % Get the axes of the original figure
    originalAxes = gca;

    originalAxesPosition = get(originalAxes, 'Position'); 

    % Create new axes for the inset plot, positioned in the middle of the figure
    % The 'Position' vector is [left bottom width height]
    insetAxes = axes('Position',[0.45 0.45 0.3 0.3]); % Adjust these values as necessary

    % Copy the contents from the original axes to the inset axes
    % This method assumes a simple plot; complex figures may require a different approach
    copyobj(allchild(originalAxes), insetAxes);

    % Set the axes limits if you want to focus on a specific region or "zoom" differently
    xlim(insetAxes, [250000 300000]);
    % ylim(insetAxes, [minY maxY]);

    % Remove all ticks from the x and y axes in the inset plot
    set(insetAxes, 'XTick', [], 'YTick', []);

    % Turn on the box and modify tick marks to visually differentiate the auxiliary coordinates
    set(insetAxes, 'Box', 'on', 'XColor', 'black', 'YColor', 'black'); % Sets the box on and auxiliary axis colors to red

    % Add an arrow which start from the bottom right corner of the original
    % paper to the copied figure
    % Original figure spans the full window; bottom right corner is at (1,0) in normalized units.
    % If it doesn't span the full window, adjust xOrig and yOrig accordingly.
    xOrigEnd = 0.904;
    % xOrigEnd = originalAxesPosition(1) + originalAxesPosition(3); % Calculate the x position for bottom right corner
    yOrigEnd = 0.157;
    % For the inset figure, you calculate the bottom right corner based on your position vector
    % Assuming insetAxes.Position = [left bottom width height];
    insetPosition = get(insetAxes, 'Position'); % If you already know it, you can skip getting it
    xInsetEnd = insetPosition(1) + insetPosition(3); % left + width
    yInsetEnd = insetPosition(2); % bottom - since we want the bottom right, height does not contribute
    % Drawing an arrow from the original figure's bottom right to the inset's bottom right
    annotation('arrow', [xOrigEnd xInsetEnd], [yOrigEnd yInsetEnd], 'LineWidth', 0.5);

    % Optionally, link axes properties to original for interactive zooming synchrony
    % linkaxes([originalAxes,insetAxes],'xy');

    % Save the modified figure with the inset
    saveFilename = ['Zoomed-LegendOff-PLOJF-F', num2str(index), '-CC.fig'];
    savefig(figHandle, saveFilename);

    close(figHandle);
end