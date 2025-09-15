%% Beautify the figure after zoomed the local area (The third step of processing, if there is no need for a local zoom then it is ...
% the second step which can be done after the legend if off)
% 2025-6-13

for i = 1:29
    index = i;
    fprintf('Current index: %d\n', index); % Example action using the index
    %% 设置打开的文件名
    openFilename = ['Zoomed-LegendOff-PLOJF-F', num2str(index), '-CC.fig']; % Those need to zoom in.
    figHandle = openfig(openFilename, 'reuse');
    a=findobj(gcf); 
    allaxes=findall(a,'Type','axes');
    set(allaxes,'FontName','Times','LineWidth',1,'FontSize',13,'FontWeight','bold');
    alltext=findall(a,'Type','text');
    set(alltext,'FontName','Times','FontSize',13,'FontWeight','bold')
    set(gcf, 'PaperUnits', 'inches');
    krare=2.5;
    x_width=krare*5/3 ;
    y_width=krare*4/3;
    set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
    set(gca, ...
    'Box' , 'on' , ...
    'TickDir' , 'in' , ...
    'TickLength' , [.02 .02] , ...
    'XMinorTick' , 'on' , ...
    'YMinorTick' , 'on' , ...
    'YGrid' , 'off' , ...
    'XGrid' , 'off' , ...
    'XColor' , [.3 .3 .3], ...
    'YColor' , [.3 .3 .3], ...
    'LineWidth' , 1 );
    saveFilename = ['Beautified-PLOJF-F', num2str(index), '-CC'];
    print(saveFilename, '-dtiff', '-r300'); %<-Save as PNG with 300 DPI
    close(figHandle);
end

close all
