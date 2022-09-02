clear variables
t = zeros(3, 5);
%% States Dimension from 4 to 8
for nx = 4:8
    %% System Dimension
    nu = 3;
    nd = 3;
    ny = 3;

    Repeat_number = 10;
    %% multiple times
    for j = 1:Repeat_number
        %% Random System
        x0 = rand(nx,1)*10 - 5;
        e = eye(nx);
        A = rand(nx,nx)*4 - 2;
        A = A./(max(abs(eig(A))));  % Normolized A
        B = rand(nx, nu)*2 - 1;
        Bd = rand(nx, nd)*2 - 1;
        Cy = rand(ny, nx)*2 - 1;
        Dy = zeros(ny, nu);
        Dyd = rand(ny, nd)*2 - 1;

        %% Horizons
        Nc = 10;
        Ne = nx-1;
        N_seq = 30;
        
        %% Disturbance
        d_range_half = 0.25;
        d_upper = d_range_half * ones(nd, 1);
        d_lower = -d_upper;
        d = rand(nd, N_seq)*d_range_half*2 - d_range_half;

        %% Cost Weight
        Q = eye(nx);
        R = eye(nu);
        PN = eye(nx);
        
        %% System Constraints
        Cf = [eye(nx);zeros(nu, nx);-eye(nx);zeros(nu, nx)];
        Cf_hat = [eye(nx); -eye(nx)];
        Df = [zeros(nx, nu); eye(nu);zeros(nx, nu);-eye(nu)];
        Dfd = zeros(2*(nx+nu), nd);
        
        Cz = [Q^(1/2); zeros(nu,nx)];
        Cz_hat = PN^(1/2);
        Dz = [zeros(nx, nu); R^(1/2)];
        Dzd = zeros(nx + nu, nd);
        z_bar = zeros(nx + nu, 1);
        
        x_bound = 2000;
        xh = x_bound * ones(nx, 1);xl = -xh;
        uh = x_bound * ones(nu, 1);ul = -uh;
        xh_N = xh;xl_N = -xh_N;
        fk = [xh;uh;-xl;-ul];
        f_bar = [kron(ones(Nc, 1), fk);xh_N;-xl_N];
        
        %% System output and input
        % Output sequence
        y = zeros(ny, N_seq);
        % Input sequence
        u_out = zeros(nu, N_seq);
        
        
        %% States for three methods
        % method in chapter 3
        x1 = zeros(nx, N_seq);
        x1(:,1) = x0;
        cost1 = zeros(1, N_seq);
        % method in chapter 4.1
        x2 = zeros(nx, N_seq);
        x2(:,1) = x0;
        cost2 = zeros(1, N_seq);
        % method in chapter 4.2
        x3 = zeros(nx, N_seq);
        x3(:,1) = x0;
        cost3 = zeros(1, N_seq);
        
        %% Method 3
        %% Stacked System Implementation
        [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Nc);
        [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
        [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
        z_bold_bar = zeros(size(Cz_bold, 1), 1);
        d_pred_lower = kron(ones(Nc+1, 1), d_lower);
        d_pred_upper = kron(ones(Nc+1, 1), d_upper);
        %% Estimation parameters
        Estimation.len = 0;
        %% Iterations
        tic;
        x_lower = ones(nx, N_seq) * -inf;
        x_upper = ones(nx, N_seq) * inf;
        x_lower(:,1) = -5;
        x_upper(:,1) = 5;
        for k = 1:N_seq
            %% Time
            
            %% Output 
            dk = d(:,k);
            y(:, k) = Cy * x1(:,k) + Dyd * dk;
            %% Bounds Update
            if(k == 1)
                % Only for x0
                y_est = y(:, 1);
                y_est = y_est(:);
                y_tilde = y_est;
                d_est_lower = d_lower;
                d_est_upper = d_upper;
                Cy_bold = Cy;
                Dyd_bold = Dyd;
                % For each entry
                for i = 1:nx
                    A_new  = e(i, :);
                    % Upper bounds
                    cvx_begin sdp quiet
                    cvx_precision high
                    % CVX variables
                    variable g(1);
                    variable X(nx, nx) diagonal semidefinite;
                    variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                    variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                    % LMI
                    L11 = g+y_tilde'*Y*y_tilde+x_upper(:,1)'*X*x_lower(:,1)+d_est_upper'*D*d_est_lower;
                    L21 = -A_new'/2 - X*(x_upper(:, 1) + x_lower(:,1))/2 - Cy_bold'*Y*y_tilde;
                    L22 = X + Cy_bold'*Y*Cy_bold;
                    L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                    L32 = Dyd_bold'*Y*Cy_bold;
                    L33 = D + Dyd_bold'*Y*Dyd_bold;
                    L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                    minimize(g)
                    subject to
                    L >= 0
                    cvx_end
                    x_upper(i, 1) = min(x_upper(i,1), g);
                    % Lower Bounds
                    cvx_begin sdp quiet
                    cvx_precision high
                    % CVX variables
                    variable g(1);
                    variable X(nx, nx) diagonal semidefinite;
                    variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                    variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                    % LMI
                    L11 = -g + y_tilde'*Y*y_tilde + x_upper(:,1)'*X*x_lower(:,1) + d_est_upper'*D*d_est_lower;
                    L21 = A_new'/2 - X*(x_upper(:, 1) + x_lower(:,1))/2 - Cy_bold'*Y*y_tilde;
                    L22 = X + Cy_bold'*Y*Cy_bold;
                    L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                    L32 = Dyd_bold'*Y*Cy_bold;
                    L33 = D + Dyd_bold'*Y*Dyd_bold;
                    L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                    maximize(g)
                    subject to
                    L >= 0
                    cvx_end
                    x_lower(i, 1) = max(x_lower(i,1), g);
                end
            else
                % For x1 and so on
                k_start = max(1, k - Ne);
                N_e = k - k_start;
                y_est = y(:, k_start:k);
                y_est = y_est(:);
                u_est = u_out(:, k_start:k-1);
                u_est = u_est(:);
                if(Estimation.len ~= N_e) % Only update when estimation length is different
                    Estimation.len = N_e;
                    [Estimation.A_bold, Estimation.B_bold, Estimation.Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N_e);
                    [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, N_e);
                end
                y_tilde = y_est - Dy_bold*u_est;
                d_est_lower = kron(ones(N_e + 1,1), d_lower);
                d_est_upper = kron(ones(N_e + 1,1), d_upper);
                % For each entry
                for i = 1:nx
                    A_new  = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.A_bold;
                    B_new  = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.B_bold;
                    Bd_new = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.Bd_bold;
                    % Upper bounds
                    cvx_begin sdp quiet
                    cvx_precision high
                    % CVX variables
                    variable g(1);
                    variable X(nx, nx) diagonal semidefinite;
                    variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                    variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                    % LMI
                    L11 = g-B_new*u_est+y_tilde'*Y*y_tilde+x_upper(:,k_start)'*X*x_lower(:,k_start)+d_est_upper'*D*d_est_lower;
                    L21 = -A_new'/2 - X*(x_upper(:, k_start) + x_lower(:,k_start))/2 - Cy_bold'*Y*y_tilde;
                    L22 = X + Cy_bold'*Y*Cy_bold;
                    L31 = -Bd_new'/2 - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                    L32 = Dyd_bold'*Y*Cy_bold;
                    L33 = D + Dyd_bold'*Y*Dyd_bold;
                    L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                    minimize(g)
                    subject to
                    L >= 0
                    cvx_end
                    x_upper(i, k) = min(x_upper(i,k), g);
                    % Lower bounds
                    cvx_begin sdp quiet
                    cvx_precision high
                    % CVX variables
                    variable g(1);
                    variable X(nx, nx) diagonal semidefinite;
                    variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                    variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                    % LMI
                    L11 = -g + B_new*u_est + y_tilde'*Y*y_tilde + x_upper(:,k_start)'*X*x_lower(:,k_start) + d_est_upper'*D*d_est_lower;
                    L21 = A_new'/2 - X*(x_upper(:, k_start) + x_lower(:,k_start))/2 - Cy_bold'*Y*y_tilde;
                    L22 = X + Cy_bold'*Y*Cy_bold;
                    L31 = Bd_new'/2 - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                    L32 = Dyd_bold'*Y*Cy_bold;
                    L33 = D + Dyd_bold'*Y*Dyd_bold;
                    L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                    maximize(g)
                    subject to
                    L >= 0
                    cvx_end
                    x_lower(i, k) = max(x_lower(i,k), g);
                end
            end
            %% RMPC
            x_over = x_upper(:,k);
            x_under = x_lower(:,k);
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variables g(1) m(1);
            variable u(Nc*nu ,1);
            variable X(nx, nx)  diagonal semidefinite;
            variable X_check(nx, nx)  diagonal semidefinite;
            variable D(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
            variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
            variable M(size(Cf_bold, 1), size(Cf_bold, 1))  diagonal;
            % LMIs
            L11 = Dz_bold'*Dz_bold;
            L21 = u'*(Dz_bold'*Dz_bold);
            L22 = g - z_bold_bar'*z_bold_bar + 2*z_bold_bar'*Dz_bold*u + x_over'*X*x_under+d_pred_upper'*D*d_pred_lower;
            L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
            L32 = Cz_bold'*(z_bold_bar - Dz_bold*u) - X*(x_over+x_under)/2;
            L33 = X - Cz_bold'*Cz_bold;
            L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
            L42 = Dzd_bold'*(z_bold_bar - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2;
            L43 = -Dzd_bold'*Cz_bold;
            L44 = D - Dzd_bold'*Dzd_bold;
            L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
            L_11 = 2*m + x_over'*X_check*x_under + d_pred_upper'*D_check*d_pred_lower;
            L_21 = f_bar - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
            L_22 = 2*M;
            L_31 = -X_check*(x_over+x_under)/2;
            L_32 = -Cf_bold';
            L_33 = X_check;
            L_41 = -D_check*(d_pred_lower+d_pred_upper)/2;
            L_42 = -Dfd_bold';
            L_43 = zeros(size(D_check, 1), size(X_check, 2));
            L_44 = D_check;
            L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
            minimize(g)
            subject to
            L >= 0
            L_ >= 0
            cvx_end
            u0 = u(1:nu);
            z = Cz * x1(:,k) + Dz * u0 + Dzd * dk;
            cost1(1, k) = z'*z;
            x1(:, k+1) = A*x1(:,k) + B*u0 + Bd*dk;
            u_out(:,k) = u0;
            
        end
        %% Time
        t(1, nx - 3) = t(1, nx - 3) + toc;
        %% Stacked System Implementation
        [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
        [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
        z_bold_bar = zeros(size(Cz_bold, 1), 1);
        
        %% Iterations with method 4.1
        %% Time
        tic;
        x_lower = -5 *ones(nx, N_seq+1);
        x_upper = 5* ones(nx, N_seq+1);
        for k = 1:N_seq
            
            %% Output 
            dk = d(:,k);
            y(:, k) = Cy * x2(:,k) + Dyd * dk;
            k_start = 1;
            N_e = k - k_start;
            y_est = y(:, k_start:k);
            y_est = y_est(:);
            d_pred_lower = kron(ones(Nc+N_e+1, 1), d_lower);
            d_pred_upper = kron(ones(Nc+N_e+1, 1), d_upper);
            % First method
            d_est_lower = kron(ones(N_e + 1,1), d_lower);
            d_est_upper = kron(ones(N_e + 1,1), d_upper);
            if(k == 1)
                % Only for x0
                y_tilde = y_est;
                Cy_bold = Cy;
                Dyd_bold = Dyd;
                f0 = f_bar;
                z0 = z_bold_bar;
            else
                % For x1 and so on
                %% Stacked System Implementation
                [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
                [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
                [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N_e);
                [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, N_e);
                u_est = u_out(:, k_start:k-1);
                u_est = u_est(:);
                y_tilde = y_est - Dy_bold*u_est;
                f0 = f_bar - Cf_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
                z0 = z_bold_bar - Cz_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
                tmp = [zeros(nx, nx*(N_e - 1)) eye(nx)] * Bd_bold;
                tmp = tmp(:, 1:end - nd);
                Dfd_bold = [Cf_bold*tmp Dfd_bold];
                Dzd_bold = [Cz_bold*tmp Dzd_bold];
                Cf_bold = Cf_bold * A^N_e;
                Cz_bold = Cz_bold * A^N_e;
            end
            % Bounds Estimation
            for i = 1:nx
                % Upper bounds
                cvx_begin sdp quiet
                cvx_precision high
                % CVX variables
                variable g(1);
                variable X(nx, nx) diagonal semidefinite;
                variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                % LMI
                L11 = g+y_tilde'*Y*y_tilde+x_upper(:,k)'*X*x_lower(:,k)+d_est_upper'*D*d_est_lower;
                L21 = -e(:,i)/2 - X*(x_upper(:,k) + x_lower(:,k))/2 - Cy_bold'*Y*y_tilde;
                L22 = X + Cy_bold'*Y*Cy_bold;
                L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                L32 = Dyd_bold'*Y*Cy_bold;
                L33 = D + Dyd_bold'*Y*Dyd_bold;
                L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                minimize(g)
                subject to
                L >= 0
                cvx_end
                x_upper(i,k+1) = min(x_upper(i,k+1), g);
                % Lower Bounds
                cvx_begin sdp quiet
                cvx_precision high
                % CVX variables
                variable g(1);
                variable X(nx, nx) diagonal semidefinite;
                variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
                % LMI
                L11 = -g + y_tilde'*Y*y_tilde + x_upper(:,k)'*X*x_lower(:,k) + d_est_upper'*D*d_est_lower;
                L21 = e(:,i)/2 - X*(x_upper(:,k) + x_lower(:,k))/2 - Cy_bold'*Y*y_tilde;
                L22 = X + Cy_bold'*Y*Cy_bold;
                L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
                L32 = Dyd_bold'*Y*Cy_bold;
                L33 = D + Dyd_bold'*Y*Dyd_bold;
                L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
                maximize(g)
                subject to
                L >= 0
                cvx_end
                x_lower(i,k+1) = max(x_lower(i,k+1), g);
            end
            % LMI
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variables g(1) m(1);
            variable u(Nc*nu ,1);
            variable X(nx, nx)  diagonal semidefinite;
            variable X_check(nx, nx)  diagonal semidefinite;
            variable D(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
            variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
            variable M(size(Cf_bold, 1), size(Cf_bold, 1))  diagonal;
            % LMIs
            L11 = Dz_bold'*Dz_bold;
            L21 = u'*(Dz_bold'*Dz_bold);
            L22 = g - z0'*z0 + 2*z0'*Dz_bold*u + x_upper(:,k)'*X*x_lower(:,k)+d_pred_upper'*D*d_pred_lower;
            L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
            L32 = Cz_bold'*(z0 - Dz_bold*u) - X*(x_upper(:,k)+x_lower(:,k))/2;
            L33 = X - Cz_bold'*Cz_bold;
            L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
            L42 = Dzd_bold'*(z0 - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2;
            L43 = -Dzd_bold'*Cz_bold;
            L44 = D - Dzd_bold'*Dzd_bold;
            L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
            L_11 = 2*m + x_upper(:,k)'*X_check*x_lower(:,k) + d_pred_upper'*D_check*d_pred_lower;
            L_21 = f0 - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
            L_22 = 2*M;
            L_31 = -X_check*(x_upper(:,k)+x_lower(:,k))/2;
            L_32 = -Cf_bold';
            L_33 = X_check;
            L_41 = -D_check*(d_pred_lower+d_pred_upper)/2;
            L_42 = -Dfd_bold';
            L_43 = zeros(size(D_check, 1), size(X_check, 2));
            L_44 = D_check;
            L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
            minimize(g)
            subject to
            L >= 0
            L_ >= 0
            cvx_end
            u0 = u(1:nu);
            z = Cz * x2(:,k) + Dz * u0 + Dzd * dk;
            cost2(1, k) = z'*z;
            x2(:, k+1) = A*x2(:,k) + B*u0 + Bd*dk;
            u_out(:,k) = u0;
        end
        %% Time
        t(2, nx - 3) = t(2, nx - 3) + toc;
        %% Iterations with method 4.2
        %% Time
        tic;
        x_lower = -5 *ones(nx, 1);
        x_upper = 5* ones(nx, 1);
        for k = 1:N_seq
            %% Output 
            dk = d(:,k);
            y(:, k) = Cy * x3(:,k) + Dyd * dk;
            k_start = max(1, k - Ne);
            N_e = k - k_start;
            y_est = y(:, k_start:k);
            y_est = y_est(:);
            d_pred_lower = kron(ones(Nc+N_e+1, 1), d_lower);
            d_pred_upper = kron(ones(Nc+N_e+1, 1), d_upper);
            if(N_e ~= Ne)
                %% Stacked System Implementation
                [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
                [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
                if(N_e == 0)
                    y_tilde = y_est;
                    Cy_bold = Cy;
                    Dyd_bold = Dyd;
                    f0 = f_bar;
                    z0 = z_bold_bar;
                else
                    [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N_e);
                    [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, N_e);
                    u_est = u_out(:, k_start:k-1);
                    u_est = u_est(:);
                    y_tilde = y_est - Dy_bold*u_est;
                    f0 = f_bar - Cf_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
                    z0 = z_bold_bar - Cz_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
                    tmp = [zeros(nx, nx*(N_e - 1)) eye(nx)] * Bd_bold;
                    tmp = tmp(:, 1:end - nd);
                    Dfd_bold = [Cf_bold*tmp Dfd_bold];
                    Dzd_bold = [Cz_bold*tmp Dzd_bold];
                    Cf_bold = Cf_bold * A^N_e;
                    Cz_bold = Cz_bold * A^N_e;
                end
                Dyd_bold = [Dyd_bold zeros(size(Dyd_bold, 1), nd*Nc)];
                % LMI
                cvx_begin sdp quiet
                cvx_precision high
                % CVX variables
                variables g(1) m(1);
                variable u(Nc*nu ,1);
                variable X(nx, nx) diagonal semidefinite;
                variable X_check(nx, nx) diagonal semidefinite;
                variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                variable Y_check(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
                variable D(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
                variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
                variable M(size(Cf_bold, 1), size(Cf_bold, 1)) diagonal;
                % LMIs
                L11 = Dz_bold'*Dz_bold;
                L21 = u'*(Dz_bold'*Dz_bold);
                L22 = g - z0'*z0 + 2*z0'*Dz_bold*u + y_tilde'*Y*y_tilde + d_pred_upper'*D*d_pred_lower + x_upper'*X*x_lower;
                L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
                L32 = Cz_bold'*(z0 - Dz_bold*u) - X*(x_lower+x_upper)/2 - Cy_bold'*Y*y_tilde;
                L33 = X - Cz_bold'*Cz_bold + Cy_bold'*Y*Cy_bold;
                L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
                L42 = Dzd_bold'*(z0 - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2 - Dyd_bold'*Y*y_tilde;
                L43 = -Dzd_bold'*Cz_bold + Dyd_bold'*Y*Cy_bold;
                L44 = D - Dzd_bold'*Dzd_bold + Dyd_bold'*Y*Dyd_bold;
                L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
                L_11 = 2*m + y_tilde'*Y_check*y_tilde + d_pred_upper'*D_check*d_pred_lower + x_upper'*X_check*x_lower;
                L_21 = f0 - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
                L_22 = 2*M;
                L_31 = -Cy_bold'*Y_check*y_tilde - X_check*(x_lower+x_upper)/2;
                L_32 = -Cf_bold';
                L_33 = X_check - Cy_bold'*Y_check*Cy_bold;
                L_41 = -Dyd_bold'*Y_check*y_tilde - D_check*(d_pred_lower+d_pred_upper)/2;
                L_42 = -Dfd_bold';
                L_43 = Dyd_bold'*Y_check*Cy_bold;
                L_44 = D_check;
                L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
                minimize(g)
                subject to
                L >= 0
                L_ >= 0
                cvx_end
            else
                % Second method
                u_est = u_out(:, k_start:k-1);
                u_est = u_est(:);
                % estimation horizon is long enough
                if(N_e == Ne)
                    [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
                    [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
                    [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Ne);
                    [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, Ne);
                    f0 = f_bar - Cf_bold * [zeros(nx, nx*(Ne - 1)) eye(nx)] * B_bold *u_est;
                    z0 = z_bold_bar - Cz_bold * [zeros(nx, nx*(Ne - 1)) eye(nx)] * B_bold *u_est;
                    tmp = [zeros(nx, nx*(Ne - 1)) eye(nx)] * Bd_bold;
                    tmp = tmp(:, 1:end - nd);
                    Dfd_bold = [Cf_bold*tmp Dfd_bold];
                    Dzd_bold = [Cz_bold*tmp Dzd_bold];
                    Cf_bold = Cf_bold * A^Ne;
                    Cz_bold = Cz_bold * A^Ne;
                    Dyd_bold = [Dyd_bold zeros(size(Dyd_bold, 1), nd*Nc)];
                end
                y_tilde = y_est - Dy_bold*u_est;
                % LMI
                cvx_begin sdp quiet
                cvx_precision high
                % CVX variables
                variables g(1) m(1);
                variable u(Nc*nu ,1);
                variable Y(size(y_tilde, 1), size(y_tilde, 1))  diagonal semidefinite;
                variable Y_check(size(y_tilde, 1), size(y_tilde, 1))  diagonal semidefinite;
                variable D(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
                variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
                variable M(size(Cf_bold, 1), size(Cf_bold, 1)) diagonal;
                % LMIs
                L11 = Dz_bold'*Dz_bold;
                L21 = u'*(Dz_bold'*Dz_bold);
                L22 = g - z0'*z0 + 2*z0'*Dz_bold*u + y_tilde'*Y*y_tilde + d_pred_upper'*D*d_pred_lower;
                L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
                L32 = Cz_bold'*(z0 - Dz_bold*u) - Cy_bold'*Y*y_tilde;
                L33 = -Cz_bold'*Cz_bold + Cy_bold'*Y*Cy_bold;
                L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
                L42 = Dzd_bold'*(z0 - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2 - Dyd_bold'*Y*y_tilde;
                L43 = -Dzd_bold'*Cz_bold + Dyd_bold'*Y*Cy_bold;
                L44 = D - Dzd_bold'*Dzd_bold + Dyd_bold'*Y*Dyd_bold;
                L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
                L_11 = 2*m + y_tilde'*Y_check*y_tilde + d_pred_upper'*D_check*d_pred_lower;
                L_21 = f0 - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
                L_22 = 2*M;
                L_31 = -Cy_bold'*Y_check*y_tilde;
                L_32 = -Cf_bold';
                L_33 = Cy_bold'*Y_check*Cy_bold;
                L_41 = -Dyd_bold'*Y_check*y_tilde-D_check*(d_pred_lower+d_pred_upper)/2;
                L_42 = -Dfd_bold';
                L_43 = Dyd_bold'*Y_check*Cy_bold;
                L_44 = D_check;
                L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
                minimize(g)
                subject to
                L >= 0
                L_ >= 0
                cvx_end
            end
            u0 = u(1:nu);
            z = Cz * x3(:,k) + Dz * u0 + Dzd * dk;
            cost3(1, k) = z'*z;
            x3(:, k+1) = A*x3(:,k) + B*u0 + Bd*dk;
            u_out(:,k) = u0;
        end
        %% Time
        t(3, nx - 3) = t(3, nx - 3) + toc;
    end
end
%% Time
t = t./Repeat_number;
t