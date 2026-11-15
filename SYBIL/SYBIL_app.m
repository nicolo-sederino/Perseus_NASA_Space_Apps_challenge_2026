function SYBIL_App
%% NASA SPACE APPS - SYBIL AI: PLANETARY ANALOGUE CONTROL CENTER (PRO UI EDITION)
% Integra: PCA/SVD + K-Means (k=5) + Dual-Metric Matching + Radar Chart + Multi-Test Mission Planner
% Miglioramenti UI: 
% - Mappe invertite (3: Terra a sinistra, 4: Pianeta a destra)
% - Cubo 3D calibrato nel pannello per piena visibilità delle etichette degli assi
% - Legende nascoste di default e attivabili/disattivabili cliccando sul bottone "Legend" o sul grafico

clc; close all;

meta_cols = {'Site_Name', 'Country', 'Category', 'Latitude', 'Longitude', ...
             'Region_State', 'Geological_Setting', 'Best_Lunar_Match', ...
             'Primary_Use', 'Analog_Category', 'Operational_Status', 'Dominant_Mineral'};

%% ========================================================================================
%% 1. CARICAMENTO DATI E CALCOLO MODELLI AI (LUNA & MARTE)
%% ========================================================================================
fprintf('Avvio SYBIL AI Control Center: caricamento cataloghi e addestramento modelli PCA/K-Means...\n');

% --- LUNA ---
lunar_csv      = 'Lunar_Targets_Catalog.csv';
earth_luna_csv = 'Earth_Lunar_Analogs_Catalog.csv';
if ~isfile(lunar_csv) || ~isfile(earth_luna_csv)
    error('File CSV lunari mancanti nella cartella corrente!');
end

T_moon  = readtable(lunar_csv);
T_earth = readtable(earth_luna_csv);

moonData.names         = T_moon.Site_Name;
moonData.coords_P      = [T_moon.Latitude, T_moon.Longitude];
moonData.earth_names   = T_earth.Site_Name;
moonData.earth_country = T_earth.Country;
moonData.coords_E      = [T_earth.Latitude, T_earth.Longitude];
moonData.T_earth       = T_earth;

if ismember('Category', T_earth.Properties.VariableNames)
    moonData.earth_cat = T_earth.Category;
elseif ismember('Analog_Category', T_earth.Properties.VariableNames)
    moonData.earth_cat = T_earth.Analog_Category;
elseif ismember('Geological_Setting', T_earth.Properties.VariableNames)
    moonData.earth_cat = T_earth.Geological_Setting;
else
    moonData.earth_cat = repmat({'Lunar Analog'}, height(T_earth), 1);
end

moonData.feats  = setdiff(intersect(T_moon.Properties.VariableNames, T_earth.Properties.VariableNames, 'stable'), meta_cols, 'stable');
moonData.labels = strrep(moonData.feats, '_', ' ');
moonData.X_P    = table2array(T_moon(:, moonData.feats));
moonData.X_E    = table2array(T_earth(:, moonData.feats));
moonData.N_P    = size(moonData.X_P, 1);
moonData.N_E    = size(moonData.X_E, 1);

% PCA & Z-Score Luna
all_l = [moonData.X_E; moonData.X_P];
mu_l = mean(all_l, 1); sd_l = std(all_l, 0, 1); sd_l(sd_l == 0) = 1e-5;
all_l_norm = (all_l - mu_l) ./ sd_l;
X_E_l_norm = all_l_norm(1:moonData.N_E, :);

[~, S_l, V_l] = svd(all_l_norm - mean(all_l_norm, 1), 'econ');
score_l = all_l_norm * V_l;
moonData.score_E_full = score_l(1:moonData.N_E, :);
moonData.score_P_full = score_l(moonData.N_E+1:end, :);
moonData.score_E_3D   = score_l(1:moonData.N_E, 1:3);
moonData.score_P_3D   = score_l(moonData.N_E+1:end, 1:3);
eig_l = diag(S_l).^2 / (size(all_l_norm, 1) - 1);
moonData.explained = (eig_l / sum(eig_l)) * 100;

% K-Means (k=5) Luna
moonData.k = 5;
moonData.cluster_idx = runKMeans(X_E_l_norm, moonData.k);

% Matching Bidirezionale Luna
[moonData.p2e_best, moonData.p2e_dist] = computeMatching(moonData.score_P_full, moonData.score_E_full);
[moonData.e2p_best, moonData.e2p_dist] = computeMatching(moonData.score_E_full, moonData.score_P_full);

% Radar Scaling Luna (congiunto)
min_l = min(all_l, [], 1); max_l = max(all_l, [], 1);
span_l = max_l - min_l; span_l(span_l == 0) = 1;
moonData.X_P_rad = (moonData.X_P - min_l) ./ span_l;
moonData.X_E_rad = (moonData.X_E - min_l) ./ span_l;

% Immagine Mappa Luna
moonData.img = loadPlanetMap('moon_map.jpg', ...
    'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg', true);

moonData.testNames = { ...
    '1 - Landing / Touchdown', ...
    '2 - Rover Mobility', ...
    '3 - Full Lunar Base', ...
    '4 - Single Habitat (2-4 Crew)', ...
    '5 - EVA / Lava Tubes', ...
    '6 - Power Systems & ISRU'};

% --- MARTE ---
mars_csv       = 'Mars_Targets_Catalog.csv';
earth_mars_csv = 'Earth_Mars_Analogs_Catalog.csv';
if ~isfile(mars_csv) || ~isfile(earth_mars_csv)
    error('File CSV marziani mancanti nella cartella corrente!');
end

T_mars       = readtable(mars_csv);
T_mars_earth = readtable(earth_mars_csv);

marsData.names         = T_mars.Site_Name;
marsData.coords_P      = [T_mars.Latitude, T_mars.Longitude];
marsData.earth_names   = T_mars_earth.Site_Name;
marsData.earth_country = T_mars_earth.Country;
marsData.earth_cat     = T_mars_earth.Category;
marsData.coords_E      = [T_mars_earth.Latitude, T_mars_earth.Longitude];
marsData.T_earth       = T_mars_earth;

marsData.feats  = setdiff(intersect(T_mars.Properties.VariableNames, T_mars_earth.Properties.VariableNames, 'stable'), meta_cols, 'stable');
marsData.labels = strrep(marsData.feats, '_', ' ');
marsData.X_P    = table2array(T_mars(:, marsData.feats));
marsData.X_E    = table2array(T_mars_earth(:, marsData.feats));
marsData.N_P    = size(marsData.X_P, 1);
marsData.N_E    = size(marsData.X_E, 1);

% PCA & Z-Score Marte
all_m = [marsData.X_E; marsData.X_P];
mu_m = mean(all_m, 1); sd_m = std(all_m, 0, 1); sd_m(sd_m == 0) = 1e-5;
all_m_norm = (all_m - mu_m) ./ sd_m;
X_E_m_norm = all_m_norm(1:marsData.N_E, :);

[~, S_m, V_m] = svd(all_m_norm - mean(all_m_norm, 1), 'econ');
score_m = all_m_norm * V_m;
marsData.score_E_full = score_m(1:marsData.N_E, :);
marsData.score_P_full = score_m(marsData.N_E+1:end, :);
marsData.score_E_3D   = score_m(1:marsData.N_E, 1:3);
marsData.score_P_3D   = score_m(marsData.N_E+1:end, 1:3);
eig_m = diag(S_m).^2 / (size(all_m_norm, 1) - 1);
marsData.explained = (eig_m / sum(eig_m)) * 100;

% K-Means (k=5) Marte
marsData.k = 5;
marsData.cluster_idx = runKMeans(X_E_m_norm, marsData.k);

% Matching Bidirezionale Marte
[marsData.p2e_best, marsData.p2e_dist] = computeMatching(marsData.score_P_full, marsData.score_E_full);
[marsData.e2p_best, marsData.e2p_dist] = computeMatching(marsData.score_E_full, marsData.score_P_full);

% Radar Scaling Marte (Intra-dominio)
min_em = min(marsData.X_E, [], 1); max_em = max(marsData.X_E, [], 1);
span_em = max_em - min_em; span_em(span_em == 0) = 1;
marsData.X_E_rad = (marsData.X_E - min_em) ./ span_em;

min_mm = min(marsData.X_P, [], 1); max_mm = max(marsData.X_P, [], 1);
span_mm = max_mm - min_mm; span_mm(span_mm == 0) = 1;
marsData.X_P_rad = (marsData.X_P - min_mm) ./ span_mm;

% Immagine Mappa Marte
marsData.img = loadPlanetMap('mars_map.jpg', ...
    'https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/1280px-OSIRIS_Mars_true_color.jpg', false);

marsData.testNames = { ...
    '1 - Landing / EDL', ...
    '2 - Rover Mobility', ...
    '3 - Full Martian Base', ...
    '4 - Single Habitat (2-4 Crew)', ...
    '5 - Drone / UAV Testing', ...
    '6 - Extreme EVA Testing', ...
    '7 - Power Systems & ISRU'};

%% ========================================================================================
%% 2. COSTRUZIONE INTERFACCIA GRAFICA APP DESIGNER (uifigure)
%% ========================================================================================
app.State.Planet    = 'MOON';
app.State.TargetID  = 1;
app.State.TestLayer = gobjects(0);
app.Lgd3D           = gobjects(0);
app.LgdRadar        = gobjects(0);
app.LgdEarth        = gobjects(0);
app.LgdPlanet       = gobjects(0);

app.Fig = uifigure('Name', 'SYBIL AI - NASA Planetary Analogue Control Center', ...
    'Position', [30, 30, 1500, 880], 'Color', [0.09 0.10 0.13]);

% Griglia Principale: Colonna sinistra (Controlli 330px) + Colonna destra (Dashboard 1x)
mainGrid = uigridlayout(app.Fig, [1, 2]);
mainGrid.ColumnWidth = {330, '1x'};
mainGrid.Padding = [10 10 10 10];
mainGrid.ColumnSpacing = 10;
mainGrid.BackgroundColor = [0.09 0.10 0.13];

% --- PANNELLO SINISTRO: SYBIL AI ---
ctrlPanel = uipanel(mainGrid, 'Title', 'SYBIL AI', ...
    'BackgroundColor', [0.13 0.15 0.19], 'ForegroundColor', [0.35 0.85 1.0], ...
    'FontSize', 16, 'FontWeight', 'bold');

ctrlGrid = uigridlayout(ctrlPanel, [10, 1]);
ctrlGrid.RowHeight = {22, 34, 22, 30, 155, 22, 165, 34, 30, '1x'};
ctrlGrid.Padding = [10 10 10 10];
ctrlGrid.RowSpacing = 6;
ctrlGrid.BackgroundColor = [0.13 0.15 0.19];

% 1. Selettore Pianeta
uilabel(ctrlGrid, 'Text', '1. CELESTIAL BODY:', 'FontColor', [0.85 0.88 0.92], 'FontWeight', 'bold', 'FontSize', 11);
app.PlanetSwitch = uibuttongroup(ctrlGrid, 'BackgroundColor', [0.18 0.20 0.25], 'BorderType', 'none', ...
    'SelectionChangedFcn', @(src, evt) onPlanetChanged(evt));
uitogglebutton(app.PlanetSwitch, 'Text', 'MOON (24 Feat.)', 'Position', [4 4 140 26], ...
    'BackgroundColor', [0.22 0.25 0.32], 'FontColor', 'w', 'FontWeight', 'bold', 'Value', true);
uitogglebutton(app.PlanetSwitch, 'Text', 'MARS (15 Feat.)', 'Position', [148 4 140 26], ...
    'BackgroundColor', [0.22 0.25 0.32], 'FontColor', 'w', 'FontWeight', 'bold');

% 2. Selettore Target Planetario
uilabel(ctrlGrid, 'Text', '2. TARGET SITE (Or click on maps/3D):', 'FontColor', [0.85 0.88 0.92], 'FontWeight', 'bold', 'FontSize', 11);
app.TargetDD = uidropdown(ctrlGrid, 'Items', moonData.names, 'Value', moonData.names{1}, ...
    'BackgroundColor', [0.18 0.20 0.25], 'FontColor', 'w', 'FontSize', 10.5, ...
    'ValueChangedFcn', @(dd, ~) onTargetDropdownChanged(dd));

% 3. Scheda Telemetrica Match AI
app.MatchInfoBox = uitextarea(ctrlGrid, 'Editable', 'off', ...
    'BackgroundColor', [0.08 0.09 0.11], 'FontColor', [0.92 0.95 0.98], 'FontSize', 10.5);

% 4. Multi-Test Mission Planner
uilabel(ctrlGrid, 'Text', '3. MULTI-TEST PLANNER (Ctrl+Click):', 'FontColor', [0.45 0.95 0.65], 'FontWeight', 'bold', 'FontSize', 11);
app.TestListBox = uilistbox(ctrlGrid, 'Items', moonData.testNames, 'Multiselect', 'on', ...
    'Value', moonData.testNames(1), 'BackgroundColor', [0.18 0.20 0.25], 'FontColor', 'w', 'FontSize', 10.5);

app.RunTestBtn = uibutton(ctrlGrid, 'Text', 'RUN MULTI-TEST SITE FINDER', ...
    'BackgroundColor', [0.15 0.65 0.38], 'FontColor', 'w', 'FontWeight', 'bold', 'FontSize', 11, ...
    'ButtonPushedFcn', @(~, ~) onRunMultiTest());

app.ClearTestBtn = uibutton(ctrlGrid, 'Text', 'Clear Test Stars from Map', ...
    'BackgroundColor', [0.28 0.30 0.36], 'FontColor', 'w', 'FontSize', 10, ...
    'ButtonPushedFcn', @(~, ~) onClearTests());

% 5. Output Log Classifica Test
app.TestLogBox = uitextarea(ctrlGrid, 'Editable', 'off', ...
    'Value', {'Select one or more mission tests above and click RUN to highlight optimal Earth sites on the map.'}, ...
    'BackgroundColor', [0.08 0.09 0.11], 'FontColor', [0.75 0.95 0.82], 'FontSize', 10);


% --- PANNELLO DESTRO: DASHBOARD GRAFICA (2 RIGHE x 2 COLONNE) ---
dashGrid = uigridlayout(mainGrid, [2, 2]);
dashGrid.RowHeight = {'1.08x', '0.92x'};
dashGrid.ColumnWidth = {'1.05x', '0.95x'};
dashGrid.Padding = [0 0 0 0];
dashGrid.RowSpacing = 8;
dashGrid.ColumnSpacing = 8;
dashGrid.BackgroundColor = [0.09 0.10 0.13];

% Pannello 1: Cubo 3D PCA (Calibrato con margini per leggere PC1, PC2, PC3)
p1 = uipanel(dashGrid, 'Title', '1. CUBIC 3D PCA LATENT SPACE & K-MEANS CLUSTERS', ...
    'BackgroundColor', [0.13 0.15 0.19], 'ForegroundColor', 'w', 'FontWeight', 'bold');
app.Ax3D = uiaxes(p1, 'Units', 'normalized', 'Position', [0.11 0.13 0.78 0.78]);
app.Ax3D.Color = [0.10 0.11 0.14];
app.Ax3D.XColor = 'w'; app.Ax3D.YColor = 'w'; app.Ax3D.ZColor = 'w';
app.Ax3D.GridColor = [0.45 0.50 0.55];
addLegendButton(p1, '3D');

% Pannello 2: Radar Chart (Spider Plot)
p2 = uipanel(dashGrid, 'Title', '2. COMPARATIVE GEOPHYSICAL SIGNATURE (RADAR CHART)', ...
    'BackgroundColor', [0.13 0.15 0.19], 'ForegroundColor', 'w', 'FontWeight', 'bold');
app.AxRadar = polaraxes(p2, 'Units', 'normalized', 'Position', [0.12 0.12 0.76 0.76]);
app.AxRadar.Color = [0.10 0.11 0.14];
app.AxRadar.ThetaColor = [0.9 0.9 0.9];
app.AxRadar.RColor = [0.75 0.80 0.85];
app.AxRadar.GridColor = [0.45 0.50 0.55];
addLegendButton(p2, 'RADAR');

% Pannello 3 (INVERTITO -> Ora a Sinistra): Mappa Topografica Terrestre (GeographicAxes)
p3 = uipanel(dashGrid, 'Title', '3. EARTH ANALOGS & MISSION TEST SITES (Click sites for info)', ...
    'BackgroundColor', [0.13 0.15 0.19], 'ForegroundColor', 'w', 'FontWeight', 'bold');
app.AxEarthMap = geoaxes(p3, 'Units', 'normalized', 'Position', [0.10 0.12 0.87 0.83]);
app.AxEarthMap.AxisColor = 'w';
addLegendButton(p3, 'EARTH');

% Pannello 4 (INVERTITO -> Ora a Destra): Mappa Planetaria 2D
p4 = uipanel(dashGrid, 'Title', '4. PLANETARY SURFACE TARGET MAP (Click sites for info)', ...
    'BackgroundColor', [0.13 0.15 0.19], 'ForegroundColor', 'w', 'FontWeight', 'bold');
app.AxPlanetMap = uiaxes(p4, 'Units', 'normalized', 'Position', [0.06 0.12 0.91 0.83]);
app.AxPlanetMap.Color = [0.10 0.11 0.14];
app.AxPlanetMap.XColor = 'w'; app.AxPlanetMap.YColor = 'w';
addLegendButton(p4, 'PLANET');

% Avvio Render Iniziale (LUNA)
renderPlanetScene();
fprintf('SYBIL AI Control Center pronto!\n');

%% ========================================================================================
%% 3. CALLBACK E FUNZIONI DI RENDERING INTERATTIVE
%% ========================================================================================
    function addLegendButton(parentPanel, tagStr)
        btn = uibutton(parentPanel, 'Text', 'Legend', ...
            'Tag', tagStr, ...
            'BackgroundColor', [0.20 0.24 0.30], 'FontColor', [0.75 0.90 1.0], ...
            'FontSize', 9, 'FontWeight', 'bold', ...
            'Tooltip', 'Mostra / Nascondi legenda', ...
            'ButtonPushedFcn', @(src, ~) onToggleLegendTag(src.Tag));
        btn.Position = [10 10 56 20];
    end

    function onToggleLegendTag(tagStr)
        switch tagStr
            case '3D'
                toggleOneLegend(app.Lgd3D);
            case 'RADAR'
                toggleOneLegend(app.LgdRadar);
            case 'EARTH'
                toggleOneLegend(app.LgdEarth);
            case 'PLANET'
                toggleOneLegend(app.LgdPlanet);
        end
    end

    function toggleOneLegend(lgdHandle)
        if ~isempty(lgdHandle) && isgraphics(lgdHandle)
            if strcmp(lgdHandle.Visible, 'on')
                lgdHandle.Visible = 'off';
            else
                lgdHandle.Visible = 'on';
            end
        end
    end

    function d = getCurrentData()
        if strcmp(app.State.Planet, 'MOON')
            d = moonData;
        else
            d = marsData;
        end
    end

    function onPlanetChanged(evt)
        if contains(evt.NewValue.Text, 'MOON')
            app.State.Planet = 'MOON';
        else
            app.State.Planet = 'MARS';
        end
        app.State.TargetID = 1;
        d = getCurrentData();
        
        app.TargetDD.Items = d.names;
        app.TargetDD.Value = d.names{1};
        app.TestListBox.Items = d.testNames;
        app.TestListBox.Value = d.testNames(1);
        app.TestLogBox.Value = {'Switched celestial body. Select tests and click RUN.'};
        
        renderPlanetScene();
    end

    function onTargetDropdownChanged(dd)
        d = getCurrentData();
        idx = find(strcmp(d.names, dd.Value), 1);
        if ~isempty(idx)
            app.State.TargetID = idx;
            updateSelectedTargetView(idx, d.p2e_best(idx), d.p2e_dist(idx));
        end
    end

    function renderPlanetScene()
        d = getCurrentData();
        isMoon = strcmp(app.State.Planet, 'MOON');
        colTarget = ternary(isMoon, [1 0.88 0.1], [1 0.42 0.1]);
        
        % --- 1. CUBO 3D PCA (CALIBRATO PER LEGGERE LE SCRITTE DEGLI ASSI) ---
        cla(app.Ax3D);
        hold(app.Ax3D, 'on');
        best_unique = unique(d.p2e_best);
        rev_names = d.names(d.e2p_best);
        
        cmap = parula(d.k);
        for c = 1:d.k
            idx_c = find(d.cluster_idx == c);
            if isempty(idx_c), continue; end
            
            cats_c = strtrim(regexprep(d.earth_cat(idx_c), '\s*\(.*?\)', ''));
            [u_cats, ~, ic_cats] = unique(cats_c);
            counts = accumarray(ic_cats, 1);
            [~, s_ord] = sort(counts, 'descend');
            u_sorted = u_cats(s_ord);
            if isscalar(u_sorted)
                lbl_c = sprintf('C%d: %s', c, u_sorted{1});
            else
                lbl_c = sprintf('C%d: %s / %s', c, u_sorted{1}, u_sorted{2});
            end
            
            sc_e = scatter3(app.Ax3D, d.score_E_3D(idx_c,1), d.score_E_3D(idx_c,2), d.score_E_3D(idx_c,3), ...
                60, cmap(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5, 'DisplayName', lbl_c);
            sc_e.DataTipTemplate.DataTipRows = [ ...
                dataTipTextRow('Earth Site', d.earth_names(idx_c)), ...
                dataTipTextRow('Country', d.earth_country(idx_c)), ...
                dataTipTextRow('Cluster', repmat({lbl_c}, length(idx_c), 1)), ...
                dataTipTextRow('Best Equivalent', rev_names(idx_c))];
            set(sc_e, 'ButtonDownFcn', @(src, evt) onClickEarth(src, evt, idx_c));
        end
        
        sc_p = scatter3(app.Ax3D, d.score_P_3D(:,1), d.score_P_3D(:,2), d.score_P_3D(:,3), ...
            140, colTarget, 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', 'Planetary Targets');
        sc_b = scatter3(app.Ax3D, d.score_E_3D(best_unique,1), d.score_E_3D(best_unique,2), d.score_E_3D(best_unique,3), ...
            115, 'r', 'LineWidth', 1.8, 'DisplayName', 'Best Matches');
        
        sc_p.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Target', d.names), ...
            dataTipTextRow('Best Analog', d.earth_names(d.p2e_best)), ...
            dataTipTextRow('PCA Dist', d.p2e_dist, '%.3f')];
        set(sc_p, 'ButtonDownFcn', @(src, evt) onClickPlanet(src, evt));
        set(sc_b, 'ButtonDownFcn', @(src, evt) onClickEarth(src, evt, best_unique));
        
        all_pts = [d.score_E_3D; d.score_P_3D];
        mn = min(all_pts, [], 1); mx = max(all_pts, [], 1); pd = (mx - mn) * 0.12;
        xlim(app.Ax3D, [mn(1)-pd(1) mx(1)+pd(1)]);
        ylim(app.Ax3D, [mn(2)-pd(2) mx(2)+pd(2)]);
        zlim(app.Ax3D, [mn(3)-pd(3) mx(3)+pd(3)]);
        pbaspect(app.Ax3D, [1 1 1]); axis(app.Ax3D, 'vis3d'); box(app.Ax3D, 'on');
        grid(app.Ax3D, 'on'); view(app.Ax3D, 38, 22);
        app.Ax3D.FontSize = 8.5;
        xlabel(app.Ax3D, sprintf('PC1 (%.1f%%)', d.explained(1)), 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
        ylabel(app.Ax3D, sprintf('PC2 (%.1f%%)', d.explained(2)), 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
        zlabel(app.Ax3D, sprintf('PC3 (%.1f%%)', d.explained(3)), 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
        
        app.Lgd3D = legend(app.Ax3D, 'Location', 'northeast', 'FontSize', 7.2, 'Visible', 'off');
        app.Lgd3D.TextColor = 'w'; app.Lgd3D.Color = [0.14 0.16 0.20]; app.Lgd3D.EdgeColor = [0.35 0.40 0.45];
        app.Lgd3D.ItemHitFcn = @(src, ~) toggleOneLegend(src);
        set(app.Ax3D, 'ButtonDownFcn', @(~, ~) onToggleLegendTag('3D'));
        
        
        % --- 3. MAPPA TOPOGRAFICA TERRESTRE (PANNELLO 3 IN BASSO A SINISTRA) ---
        cla(app.AxEarthMap);
        hold(app.AxEarthMap, 'on');
        m_all = geoscatter(app.AxEarthMap, d.coords_E(:,1), d.coords_E(:,2), ...
            50, [0.25 0.28 0.32], 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.4, 'DisplayName', 'All Earth Sites');
        m_best = geoscatter(app.AxEarthMap, d.coords_E(best_unique,1), d.coords_E(best_unique,2), ...
            155, [0.95 0.15 0.15], 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.4, 'DisplayName', 'Best AI Matches');
        
        m_all.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Earth Site', d.earth_names), ...
            dataTipTextRow('Country', d.earth_country), ...
            dataTipTextRow('Latitude (°)', d.coords_E(:,1), '%.2f'), ...
            dataTipTextRow('Longitude (°)', d.coords_E(:,2), '%.2f'), ...
            dataTipTextRow('Best Equivalent', rev_names), ...
            dataTipTextRow('PCA Distance', d.e2p_dist, '%.3f')];
            
        m_best.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Best Earth Match', d.earth_names(best_unique)), ...
            dataTipTextRow('Country', d.earth_country(best_unique)), ...
            dataTipTextRow('Latitude (°)', d.coords_E(best_unique,1), '%.2f'), ...
            dataTipTextRow('Longitude (°)', d.coords_E(best_unique,2), '%.2f'), ...
            dataTipTextRow('Best Equivalent', rev_names(best_unique)), ...
            dataTipTextRow('PCA Distance', d.e2p_dist(best_unique), '%.3f')];
        
        try
            geobasemap(app.AxEarthMap, 'colorterrain');
            if isMoon, geolimits(app.AxEarthMap, [-85 82], [-170 175]);
            else, geolimits(app.AxEarthMap, [-55 82], [-145 165]); end
        catch
            geobasemap(app.AxEarthMap, 'landcover');
        end
        
        [~, sort_e_idx] = sort(d.coords_E(best_unique, 1), 'descend');
        placed_e = zeros(0, 2);
        for ii = 1:length(best_unique)
            idx_s = best_unique(sort_e_idx(ii));
            lat_e = d.coords_E(idx_s, 1);
            lon_e = d.coords_E(idx_s, 2);
            s_lbl_e = shortName(d.earth_names{idx_s});
            
            if lon_e > 115
                hAlign = 'right';
                dlon = -3.0;
            else
                hAlign = 'left';
                dlon = 3.0;
            end
            
            dlat = 2.0;
            for prev = 1:size(placed_e, 1)
                if abs(lon_e - placed_e(prev, 1)) < 32 && abs((lat_e + dlat) - placed_e(prev, 2)) < 7.5
                    dlat = dlat - 7.5;
                end
            end
            placed_e(end+1, :) = [lon_e, lat_e + dlat]; %#ok<AGROW>
            
            text(app.AxEarthMap, lat_e + dlat, lon_e + dlon, s_lbl_e, ...
                'HorizontalAlignment', hAlign, 'FontSize', 7.5, 'FontWeight', 'bold', ...
                'Color', [1 0.92 0.92], 'BackgroundColor', [0.10 0.12 0.16 0.80], ...
                'EdgeColor', [0.85 0.25 0.25], 'Margin', 1.5, 'PickableParts', 'none');
        end
        
        app.LgdEarth = legend(app.AxEarthMap, 'Location', 'northeast', 'FontSize', 8, 'Visible', 'off');
        app.LgdEarth.TextColor = 'w'; app.LgdEarth.Color = [0.12 0.14 0.18];
        app.LgdEarth.ItemHitFcn = @(src, ~) toggleOneLegend(src);
        set(app.AxEarthMap, 'ButtonDownFcn', @(~, ~) onToggleLegendTag('EARTH'));
        
        set(m_all, 'ButtonDownFcn', @(src, evt) onClickEarth(src, evt, 1:d.N_E));
        set(m_best, 'ButtonDownFcn', @(src, evt) onClickEarth(src, evt, best_unique));
        
        
        % --- 4. MAPPA PLANETARIA 2D (PANNELLO 4 IN BASSO A DESTRA) ---
        cla(app.AxPlanetMap);
        im_p = image(app.AxPlanetMap, [-180 180], [90 -90], d.img);
        set(im_p, 'PickableParts', 'none');
        set(app.AxPlanetMap, 'YDir', 'normal');
        hold(app.AxPlanetMap, 'on');
        
        p_map_pts = scatter(app.AxPlanetMap, d.coords_P(:,2), d.coords_P(:,1), ...
            180, colTarget, 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.6, 'DisplayName', 'Planetary Targets');
        
        p_map_pts.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Site Name', d.names), ...
            dataTipTextRow('Latitude (°)', d.coords_P(:,1), '%.2f'), ...
            dataTipTextRow('Longitude (°)', d.coords_P(:,2), '%.2f'), ...
            dataTipTextRow('Best Earth Analog', d.earth_names(d.p2e_best)), ...
            dataTipTextRow('PCA Distance', d.p2e_dist, '%.3f')];
        
        [~, sort_lat_idx] = sort(d.coords_P(:,1), 'descend');
        placed_boxes = zeros(0, 2);
        for ii = 1:d.N_P
            m = sort_lat_idx(ii);
            lon_m = d.coords_P(m, 2);
            lat_m = d.coords_P(m, 1);
            s_lbl = shortName(d.names{m});
            
            if lon_m > 95
                hAlign = 'right';
                dx = -4.5;
            else
                hAlign = 'left';
                dx = 4.5;
            end
            
            dy = 3.5;
            if lat_m > 75, dy = -6.0; end
            for prev = 1:size(placed_boxes, 1)
                if abs(lon_m - placed_boxes(prev, 1)) < 38 && abs((lat_m + dy) - placed_boxes(prev, 2)) < 9.5
                    dy = dy - 10.5;
                end
            end
            placed_boxes(end+1, :) = [lon_m, lat_m + dy]; %#ok<AGROW>
            
            text(app.AxPlanetMap, lon_m + dx, lat_m + dy, s_lbl, ...
                'HorizontalAlignment', hAlign, 'FontSize', 7.5, 'FontWeight', 'bold', ...
                'Color', [1 1 0.35], 'BackgroundColor', [0 0 0 0.72], ...
                'EdgeColor', [0.4 0.4 0.4], 'Margin', 1.5, 'PickableParts', 'none');
        end
        
        xlim(app.AxPlanetMap, [-180 180]); ylim(app.AxPlanetMap, [-90 90]);
        xticks(app.AxPlanetMap, -180:60:180); yticks(app.AxPlanetMap, -90:30:90);
        grid(app.AxPlanetMap, 'on'); app.AxPlanetMap.GridColor = [0.85 0.85 0.85]; app.AxPlanetMap.GridAlpha = 0.25;
        xlabel(app.AxPlanetMap, 'Longitude (°)', 'Color', 'w', 'FontWeight', 'bold');
        ylabel(app.AxPlanetMap, 'Latitude (°)', 'Color', 'w', 'FontWeight', 'bold');
        set(p_map_pts, 'ButtonDownFcn', @(src, evt) onClickPlanet(src, evt));
        
        app.LgdPlanet = legend(app.AxPlanetMap, 'Location', 'northeast', 'FontSize', 8, 'Visible', 'off');
        app.LgdPlanet.TextColor = 'w'; app.LgdPlanet.Color = [0.12 0.14 0.18];
        app.LgdPlanet.ItemHitFcn = @(src, ~) toggleOneLegend(src);
        set(app.AxPlanetMap, 'ButtonDownFcn', @(~, ~) onToggleLegendTag('PLANET'));
        
        % Aggiorna Radar e Scheda per il target attivo
        p_idx = app.State.TargetID;
        updateSelectedTargetView(p_idx, d.p2e_best(p_idx), d.p2e_dist(p_idx));
    end

    function updateSelectedTargetView(p_idx, e_idx, dist_val)
        d = getCurrentData();
        isMoon = strcmp(app.State.Planet, 'MOON');
        colP = ternary(isMoon, [1 0.85 0.1], [1 0.45 0]);
        
        app.TargetDD.Value = d.names{p_idx};
        
        app.MatchInfoBox.Value = { ...
            sprintf('[ %s TARGET ]', app.State.Planet), ...
            sprintf('Site: %s', d.names{p_idx}), ...
            sprintf('Coords: %.2f°, %.2f°', d.coords_P(p_idx,1), d.coords_P(p_idx,2)), ...
            '----------------------------------------', ...
            '[ OPTIMAL EARTH ANALOG ]', ...
            sprintf('Site: %s (%s)', d.earth_names{e_idx}, d.earth_country{e_idx}), ...
            sprintf('Category: %s', d.earth_cat{e_idx}), ...
            sprintf('GPS: %.2f°, %.2f°', d.coords_E(e_idx,1), d.coords_E(e_idx,2)), ...
            sprintf('PCA Euclidean Dist: %.3f', dist_val)};
        
        % Aggiorna Radar Chart
        cla(app.AxRadar);
        num_f = length(d.feats);
        angles_deg = linspace(0, 360, num_f + 1);
        angles_rad = deg2rad(angles_deg);
        
        rp = [d.X_P_rad(p_idx,:), d.X_P_rad(p_idx,1)];
        re = [d.X_E_rad(e_idx,:), d.X_E_rad(e_idx,1)];
        raw_p = [d.X_P(p_idx,:), d.X_P(p_idx,1)];
        raw_e = [d.X_E(e_idx,:), d.X_E(e_idx,1)];
        lbl_c = [d.labels(:)', d.labels(1)];
        
        lp = polarplot(app.AxRadar, angles_rad, rp, '-o', 'LineWidth', 2.2, ...
            'Color', colP, 'MarkerFaceColor', colP, 'MarkerSize', 5, 'DisplayName', shortName(d.names{p_idx}));
        hold(app.AxRadar, 'on');
        le = polarplot(app.AxRadar, angles_rad, re, '-s', 'LineWidth', 2.0, ...
            'Color', [1 0.25 0.25], 'MarkerFaceColor', 'r', 'MarkerSize', 5, 'DisplayName', shortName(d.earth_names{e_idx}));
        hold(app.AxRadar, 'off');
        
        lp.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Site', repmat({d.names{p_idx}}, 1, num_f+1)), ...
            dataTipTextRow('Feature', lbl_c), ...
            dataTipTextRow('Actual Value (SI)', raw_p, '%.2f'), ...
            dataTipTextRow('Normalized (%)', rp*100, '%.1f%%')];
        le.DataTipTemplate.DataTipRows = [ ...
            dataTipTextRow('Site', repmat({d.earth_names{e_idx}}, 1, num_f+1)), ...
            dataTipTextRow('Feature', lbl_c), ...
            dataTipTextRow('Actual Value (SI)', raw_e, '%.2f'), ...
            dataTipTextRow('Normalized (%)', re*100, '%.1f%%')];
        
        app.AxRadar.ThetaTick = angles_deg(1:end-1);
        app.AxRadar.ThetaTickLabel = d.labels;
        app.AxRadar.FontSize = 7.5;
        rlim(app.AxRadar, [0 1]);
        rticks(app.AxRadar, [0.25 0.5 0.75 1.0]);
        rticklabels(app.AxRadar, {'25%', '50%', '75%', '100%'});
        title(app.AxRadar, sprintf('%s  vs  %s (Dist: %.2f)', shortName(d.names{p_idx}), shortName(d.earth_names{e_idx}), dist_val), ...
            'Color', 'w', 'FontSize', 9.5, 'FontWeight', 'bold');
        
        app.LgdRadar = legend(app.AxRadar, 'Location', 'northeast', 'FontSize', 8, 'Visible', 'off');
        app.LgdRadar.TextColor = 'w'; app.LgdRadar.Color = [0.16 0.18 0.22];
        app.LgdRadar.ItemHitFcn = @(src, ~) toggleOneLegend(src);
        set(app.AxRadar, 'ButtonDownFcn', @(~, ~) onToggleLegendTag('RADAR'));
    end

    function onClickPlanet(src, evt)
        d = getCurrentData();
        p_idx = findClickedIdx(src, evt);
        app.State.TargetID = p_idx;
        updateSelectedTargetView(p_idx, d.p2e_best(p_idx), d.p2e_dist(p_idx));
    end

    function onClickEarth(src, evt, idx_map)
        d = getCurrentData();
        raw_i = findClickedIdx(src, evt);
        e_idx = idx_map(raw_i);
        p_idx = d.e2p_best(e_idx);
        app.State.TargetID = p_idx;
        updateSelectedTargetView(p_idx, e_idx, d.e2p_dist(e_idx));
    end

    function onClearTests()
        if ~isempty(app.State.TestLayer) && any(isgraphics(app.State.TestLayer))
            wasVis = 'off';
            if ~isempty(app.LgdEarth) && isgraphics(app.LgdEarth)
                wasVis = app.LgdEarth.Visible;
            end
            delete(app.State.TestLayer(isgraphics(app.State.TestLayer)));
            app.LgdEarth = legend(app.AxEarthMap, 'Location', 'northeast', 'FontSize', 8, 'Visible', wasVis);
            app.LgdEarth.TextColor = 'w'; app.LgdEarth.Color = [0.12 0.14 0.18];
            app.LgdEarth.ItemHitFcn = @(src, ~) toggleOneLegend(src);
        end
        app.TestLogBox.Value = {'Test markers cleared from Earth map.'};
    end

    function onRunMultiTest()
        d = getCurrentData();
        selectedItems = app.TestListBox.Value;
        if isempty(selectedItems)
            app.TestLogBox.Value = {'Please select at least one test from the list!'};
            return;
        end
        if ischar(selectedItems), selectedItems = {selectedItems}; end
        
        test_ids = zeros(1, length(selectedItems));
        for i = 1:length(selectedItems)
            test_ids(i) = find(strcmp(d.testNames, selectedItems{i}), 1);
        end
        test_ids = unique(test_ids, 'stable');
        num_t = length(test_ids);
        
        if strcmp(app.State.Planet, 'MOON')
            [idx_validi, num_passed, score_comb, test_nome, all_ok] = evaluateMoonTests(d.T_earth, test_ids);
            starCol = 'g';
        else
            [idx_validi, num_passed, score_comb, test_nome, all_ok] = evaluateMarsTests(d.T_earth, d.earth_cat, test_ids);
            starCol = 'c';
        end
        
        wasVis = 'off';
        if ~isempty(app.LgdEarth) && isgraphics(app.LgdEarth)
            wasVis = app.LgdEarth.Visible;
        end
        
        if ~isempty(app.State.TestLayer) && any(isgraphics(app.State.TestLayer))
            delete(app.State.TestLayer(isgraphics(app.State.TestLayer)));
        end
        
        if ~isempty(idx_validi)
            hold(app.AxEarthMap, 'on');
            rev_names = d.names(d.e2p_best);
            app.State.TestLayer = geoscatter(app.AxEarthMap, d.coords_E(idx_validi,1), d.coords_E(idx_validi,2), ...
                240, starCol, 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.3, ...
                'DisplayName', ['Test: ' test_nome]);
            
            app.State.TestLayer.DataTipTemplate.DataTipRows = [ ...
                dataTipTextRow('Test Site', d.earth_names(idx_validi)), ...
                dataTipTextRow('Country', d.earth_country(idx_validi)), ...
                dataTipTextRow('Latitude (°)', d.coords_E(idx_validi,1), '%.2f'), ...
                dataTipTextRow('Longitude (°)', d.coords_E(idx_validi,2), '%.2f'), ...
                dataTipTextRow('Best Equivalent', rev_names(idx_validi)), ...
                dataTipTextRow('Test Score', score_comb(idx_validi), '%.1f/100')];
            set(app.State.TestLayer, 'ButtonDownFcn', @(src, evt) onClickEarth(src, evt, idx_validi));
            
            app.LgdEarth = legend(app.AxEarthMap, 'Location', 'northeast', 'FontSize', 8, 'Visible', wasVis);
            app.LgdEarth.TextColor = 'w'; app.LgdEarth.Color = [0.12 0.14 0.18];
            app.LgdEarth.ItemHitFcn = @(src, ~) toggleOneLegend(src);
            
            logLines = cell(length(idx_validi) + 2, 1);
            if all_ok
                logLines{1} = sprintf('SUCCESS: %d sites match ALL %d tests:', length(idx_validi), num_t);
            else
                logLines{1} = sprintf('BEST-FIT (%d/%d tests passed):', max(num_passed), num_t);
            end
            logLines{2} = ['[' test_nome ']'];
            for k_i = 1:length(idx_validi)
                id = idx_validi(k_i);
                logLines{k_i+2} = sprintf('%d) %s (%s) - Score: %.1f/100', ...
                    k_i, d.earth_names{id}, d.earth_country{id}, score_comb(id));
            end
            app.TestLogBox.Value = logLines;
        else
            app.TestLogBox.Value = {'No matching sites found for the selected criteria.'};
        end
    end

end

%% ========================================================================================
%% FUNZIONI MATEMATICHE E DI VALUTAZIONE TEST (STANDALONE)
%% ========================================================================================
function idx = runKMeans(X_norm, k)
    N = size(X_norm, 1);
    rng(42);
    centroids = X_norm(randperm(N, k), :);
    idx = zeros(N, 1);
    for iter = 1:100
        dists = zeros(N, k);
        for c = 1:k
            dists(:, c) = sum((X_norm - centroids(c, :)).^2, 2);
        end
        [~, idx] = min(dists, [], 2);
        for c = 1:k
            pts = X_norm(idx == c, :);
            if ~isempty(pts)
                centroids(c, :) = mean(pts, 1);
            end
        end
    end
end

function [best_idx, best_dist] = computeMatching(A_full, B_full)
    N = size(A_full, 1);
    best_idx = zeros(N, 1);
    best_dist = zeros(N, 1);
    for i = 1:N
        d = sqrt(sum((B_full - A_full(i, :)).^2, 2));
        [sd, s_idx] = sort(d, 'ascend');
        best_idx(i) = s_idx(1);
        best_dist(i) = sd(1);
    end
end

function img = loadPlanetMap(localFile, urlStr, isMoon)
    if isfile(localFile)
        img = imread(localFile);
    else
        try
            img = webread(urlStr, weboptions('Timeout', 10, 'CertificateFilename', ''));
            imwrite(img, localFile);
        catch
            [Xg, Yg] = meshgrid(linspace(-180, 180, 800), linspace(-90, 90, 400));
            if isMoon
                m_s = 145 + 18 * sin(Xg * pi/65) .* cos(Yg * pi/35) + randn(400, 800) * 5;
                img = cat(3, uint8(m_s), uint8(m_s), uint8(m_s));
            else
                m_red = 175 + 25 * sin(Xg * pi/80) .* cos(Yg * pi/40) + randn(400, 800) * 4;
                img = cat(3, uint8(m_red), uint8(double(m_red)*0.52), uint8(double(m_red)*0.32));
            end
        end
    end
end

function s_short = shortName(s_full)
    idx_p = strfind(s_full, '(');
    if ~isempty(idx_p)
        s_short = strtrim(s_full(1:idx_p(1)-1));
    else
        s_short = s_full;
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function idx = findClickedIdx(src, evt)
    pt = evt.IntersectionPoint;
    if isprop(src, 'LatitudeData')
        [~, idx] = min((src.LatitudeData(:) - pt(1)).^2 + (src.LongitudeData(:) - pt(2)).^2);
    else
        xd = src.XData(:); yd = src.YData(:);
        if isprop(src, 'ZData') && ~isempty(src.ZData) && length(pt) >= 3
            zd = src.ZData(:);
            [~, idx] = min((xd - pt(1)).^2 + (yd - pt(2)).^2 + (zd - pt(3)).^2);
        else
            [~, idx] = min((xd - pt(1)).^2 + (yd - pt(2)).^2);
        end
    end
end

function [idx_validi, num_passed, score_comb, test_nome, all_ok] = evaluateMoonTests(T_earth, test_scelti)
    N_earth = height(T_earth);
    num_t = length(test_scelti);
    masks_mat  = false(N_earth, num_t);
    scores_mat = zeros(N_earth, num_t);
    nomi_scelti = cell(1, num_t);

    for k_t = 1:num_t
        switch test_scelti(k_t)
            case 1
                m_k = (T_earth.Safe_Area_m2 >= 200000) & (T_earth.Slope_5m <= 5.0) & (T_earth.Slope_1m <= 4.0) & ...
                      (T_earth.Hab_Sep_m >= 2000) & (T_earth.Boulder_Height_m <= 0.35) & ...
                      (T_earth.Boulder_Area_frac <= 0.05) & (T_earth.Bearing_Capacity_kPa >= 5.0);
                s_k = (T_earth.Safe_Area_m2 ./ max(T_earth.Safe_Area_m2)) * 30 + (1 - T_earth.Slope_5m / 10) * 25 + ...
                      (1 - T_earth.Boulder_Area_frac / 0.10) * 25 + (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 20;
                n_k = 'Landing';
            case 2
                m_k = (T_earth.Traverse_Slope_deg <= 8.0) & (T_earth.Friction_Angle_deg >= 38.0) & ...
                      (T_earth.Boulder_Height_m <= 0.38) & (T_earth.Cohesion_kPa >= 0.44) & (T_earth.Thermal_Inertia >= 200);
                s_k = (1 - T_earth.Traverse_Slope_deg / 15) * 35 + (1 - T_earth.Boulder_Height_m / 0.50) * 25 + ...
                      (T_earth.Cohesion_kPa ./ max(T_earth.Cohesion_kPa)) * 20 + (T_earth.Thermal_Inertia ./ max(T_earth.Thermal_Inertia)) * 20;
                n_k = 'Rover';
            case 3
                m_k = (T_earth.Slope_25m <= 4.6) & (T_earth.Safe_Area_m2 >= 250) & (T_earth.Bearing_Capacity_kPa >= 5.5) & ...
                      (T_earth.Sun_Visibility_pct >= 70) & (T_earth.DTE_pct >= 90) & (T_earth.Hab_Sep_m >= 2000);
                s_k = (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 30 + (T_earth.Sun_Visibility_pct / 100) * 30 + ...
                      (T_earth.Hab_Sep_m ./ max(T_earth.Hab_Sep_m)) * 25 + (1 - T_earth.Slope_25m / 10) * 15;
                n_k = 'Full Base';
            case 4
                m_k = (T_earth.Slope_25m < 7.0) & (T_earth.Safe_Area_m2 >= 25) & (T_earth.Bearing_Capacity_kPa >= 5.0) & ...
                      (T_earth.Sun_Visibility_pct >= 60) & (T_earth.DTE_pct >= 90) & (T_earth.Hab_Sep_m >= 1400);
                s_k = (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 30 + (T_earth.Sun_Visibility_pct / 100) * 30 + ...
                      (T_earth.Hab_Sep_m ./ max(T_earth.Hab_Sep_m)) * 25 + (1 - T_earth.Slope_25m / 10) * 15;
                n_k = '2-4p Habitat';
            case 5
                dT = T_earth.T_max_K - T_earth.T_min_K;
                m_k = (T_earth.Pit_Depth_m >= 20) & (T_earth.Pit_Diameter_m >= 35) & (T_earth.Cohesion_kPa >= 0.44) & (dT >= 35);
                s_k = min(T_earth.Pit_Depth_m / 50, 1) * 40 + min(dT / 60, 1) * 35 + (T_earth.Cohesion_kPa ./ max(T_earth.Cohesion_kPa)) * 25;
                n_k = 'EVA/Lava Tubes';
            case 6
                has_res = ((T_earth.FeO_wt_pct >= 13.5) & (T_earth.TiO2_wt_pct >= 3.0)) | (T_earth.Water_Ice_wt_pct > 0);
                m_k = (T_earth.Sun_Visibility_pct >= 70.0) & has_res & (T_earth.Density_0_60cm_kg_m3 <= 1710) & (T_earth.Slope_5m <= 6.5);
                res_s = max(T_earth.FeO_wt_pct / 16, T_earth.Water_Ice_wt_pct / 5);
                s_k = (T_earth.Sun_Visibility_pct / 100) * 45 + min(res_s, 1) * 35 + (1 - T_earth.Slope_5m / 10) * 20;
                n_k = 'Power/ISRU';
        end
        masks_mat(:, k_t)  = m_k;
        scores_mat(:, k_t) = s_k;
        nomi_scelti{k_t}   = n_k;
    end

    test_nome  = strjoin(nomi_scelti, ' + ');
    num_passed = sum(masks_mat, 2);
    score_comb = mean(scores_mat, 2);
    mask_all   = all(masks_mat, 2);
    all_ok     = any(mask_all);

    if all_ok
        idx_validi = find(mask_all);
        [~, ord] = sort(score_comb(idx_validi), 'descend');
        idx_validi = idx_validi(ord);
    else
        max_p = max(num_passed);
        idx_validi = find(num_passed == max_p);
        [~, ord] = sort(score_comb(idx_validi), 'descend');
        idx_validi = idx_validi(ord(1:min(5, end)));
    end
end

function [idx_validi, num_passed, score_comb, test_nome, all_ok] = evaluateMarsTests(T_mars_earth, mars_earth_category, test_scelti)
    N_mars_earth = height(T_mars_earth);
    cat_m = lower(string(mars_earth_category));
    vars_m = T_mars_earth.Properties.VariableNames;
    num_t = length(test_scelti);
    masks_mat  = false(N_mars_earth, num_t);
    scores_mat = zeros(N_mars_earth, num_t);
    nomi_scelti = cell(1, num_t);

    for k_t = 1:num_t
        m_k = true(N_mars_earth, 1);
        s_k = ones(N_mars_earth, 1) * 60;
        switch test_scelti(k_t)
            case 1
                m_k = contains(cat_m, 'desert') | contains(cat_m, 'salt') | contains(cat_m, 'plain') | contains(cat_m, 'plateau');
                if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 500000); end
                if ismember('Thermal_Inertia', vars_m), m_k = m_k & (T_mars_earth.Thermal_Inertia >= 150); s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 25; end
                if ismember('Albedo', vars_m), m_k = m_k & (T_mars_earth.Albedo <= 0.26); end
                if ismember('Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Slope_deg <= 10); s_k = s_k + (1 - T_mars_earth.Slope_deg / 15) * 15; end
                n_k = 'EDL';
            case 2
                m_k = (contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'dune') | contains(cat_m, 'plateau')) & ~contains(cat_m, 'glacio') & ~contains(cat_m, 'polar coastal');
                if ismember('Boulder_Area_frac', vars_m), m_k = m_k & (T_mars_earth.Boulder_Area_frac <= 0.07); end
                if ismember('Thermal_Inertia', vars_m), m_k = m_k & (T_mars_earth.Thermal_Inertia >= 150); s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 30; end
                if ismember('Traverse_Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Traverse_Slope_deg <= 20); end
                n_k = 'Rover';
            case 3
                m_k = (contains(cat_m, 'salt') | contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'plateau')) & ~contains(cat_m, 'mobile dunes') & ~contains(cat_m, 'glacio') & ~contains(cat_m, 'fumaroles') & ~contains(cat_m, 'high-altitude');
                if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 500); end
                if ismember('Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Slope_deg <= 5); s_k = s_k + (1 - T_mars_earth.Slope_deg / 12) * 20;
                elseif ismember('Mean_Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Mean_Slope_deg <= 5); s_k = s_k + (1 - T_mars_earth.Mean_Slope_deg / 12) * 20; end
                if ismember('Thermal_Inertia', vars_m), m_k = m_k & (T_mars_earth.Thermal_Inertia >= 180); s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 20; end
                if ismember('Bearing_Capacity_kPa', vars_m), m_k = m_k & (T_mars_earth.Bearing_Capacity_kPa >= 5.5); end
                n_k = 'Full Base';
            case 4
                m_k = (contains(cat_m, 'salt') | contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'plateau')) & ~contains(cat_m, 'mobile dunes') & ~contains(cat_m, 'glacio') & ~contains(cat_m, 'fumaroles');
                if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 40); end
                if ismember('Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Slope_deg <= 8); s_k = s_k + (1 - T_mars_earth.Slope_deg / 12) * 20;
                elseif ismember('Mean_Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Mean_Slope_deg <= 8); s_k = s_k + (1 - T_mars_earth.Mean_Slope_deg / 12) * 20; end
                if ismember('Thermal_Inertia', vars_m), m_k = m_k & (T_mars_earth.Thermal_Inertia >= 160); s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 20; end
                if ismember('Bearing_Capacity_kPa', vars_m), m_k = m_k & (T_mars_earth.Bearing_Capacity_kPa >= 4.5); end
                n_k = '2-4p Habitat';
            case 5
                m_k = contains(cat_m, 'mountain') | contains(cat_m, 'high-altitude') | contains(cat_m, 'plateau');
                if ismember('Elevation_m', vars_m), m_k = m_k & (T_mars_earth.Elevation_m >= 2500); s_k = s_k + (T_mars_earth.Elevation_m ./ max(T_mars_earth.Elevation_m)) * 35; end
                if ismember('Wind_Speed_ms', vars_m), m_k = m_k & (T_mars_earth.Wind_Speed_ms <= 10); end
                n_k = 'Drones/UAV';
            case 6
                m_k = contains(cat_m, 'cold') | contains(cat_m, 'polar') | contains(cat_m, 'arctic') | contains(cat_m, 'hyper-arid');
                if ismember('T_min_K', vars_m), m_k = m_k & (T_mars_earth.T_min_K <= 255); s_k = s_k + (1 - (T_mars_earth.T_min_K - 200) / 100) * 35; end
                if ismember('Humidity_pct', vars_m), m_k = m_k & (T_mars_earth.Humidity_pct <= 15); end
                n_k = 'Extreme EVA';
            case 7
                m_k = (contains(cat_m, 'polar') | contains(cat_m, 'periglacial') | contains(cat_m, 'salt') | contains(cat_m, 'gypsum')) & ~contains(cat_m, 'basaltic flows');
                if ismember('Water_Ice_wt_pct', vars_m), m_k = m_k | (T_mars_earth.Water_Ice_wt_pct > 0); s_k = s_k + min(T_mars_earth.Water_Ice_wt_pct / 5, 1) * 35; end
                n_k = 'ISRU/Ice';
        end
        masks_mat(:, k_t)  = m_k;
        scores_mat(:, k_t) = min(s_k, 100);
        nomi_scelti{k_t}   = n_k;
    end

    test_nome  = strjoin(nomi_scelti, ' + ');
    num_passed = sum(masks_mat, 2);
    score_comb = mean(scores_mat, 2);
    mask_all   = all(masks_mat, 2);
    all_ok     = any(mask_all);

    if all_ok
        idx_validi = find(mask_all);
        [~, ord] = sort(score_comb(idx_validi), 'descend');
        idx_validi = idx_validi(ord);
    else
        max_p = max(num_passed);
        idx_validi = find(num_passed == max_p);
        [~, ord] = sort(score_comb(idx_validi), 'descend');
        idx_validi = idx_validi(ord(1:min(5, end)));
    end
end
