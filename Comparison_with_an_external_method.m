clear variables
%% System
A   =   [-0.02 -2; 0.1 -0.7];
B_u =   [-0.5; -1];
B_d =   [0.65; -0.7];
C   =   [0.5 1];
D_d =   0.5;
%% System Dimension
n=2;nd=1;nu=1;ny=1;
du = 0.3;
dl = -du;
d = 2*du*rand(nd, 21)-du*ones(nd, 21);

%% Horizons
% Prediction Horizon
Np = 5;
Ne = 5;
% Initial Value
x0 = [5; 4];

%% Constraints
xh = 100*ones(n, 1);
xl = -xh;
xh_bar = kron(ones(Np, 1), xh);
xl_bar = -xh_bar;

XXNu(:,1) = xh;
XXnl(:,1) = xl;

%Upper and lower bounds on input signal
uh = 100;
ul = -uh;

%Array of the actual state and output of the system
X1(:, 1) = x0;
y(:, 1) = C*x0;
X_hat(:, 1)=[0; 0];
UU(:, 1) = [0];
Q = eye(Np*n);
R = eye(Np*nu);
X1(:, 2) = A*X1(:, 1) + B_u*d(:, 1)+B_u*UU(1:nu, 1);
y(3:4, 1) = C*X1(:, 2);
cost = zeros(1, 21);
tic
%% Estimation for first Ne iterations while the horizon is increasing
for N = 1:Ne
    wu = [10*ones(n, 1); kron(ones(N, 1), du*ones(nd, 1))];
    wd = -wu;
    [Phi, Gamma, Lambda] = prediction_matrices(A, B_d, B_u, N);
    I = [zeros(1, N-1), 1];
    Lambda_x = kron(I, eye(n))*Lambda;
    Gamma_x = kron(I, eye(n))*Gamma;
    Delta_x = [kron(I, eye(n))*Phi, Lambda_x];
    Lambda_y = kron(eye(N), C)*Lambda + kron(eye(N), D_d);
    Gamma_y = kron(eye(N), C)*Gamma;
    Delta_y = [kron(eye(N), C)*Phi, Lambda_y];
    e = eye(n);
    cvx_begin sdp quiet
    cvx_precision high
    variable Diu((N*nd)+n, (N*nd)+n, n) diagonal;
    variable Dil((N*nd)+n, (N*nd)+n, n) diagonal;
    variable g nonnegative;
    variable xNu(n, 1);
    variable xNl(n, 1);
    variable muu(N*ny, n);
    minimize(g)
    subject to
    [g , (xNu-xNl)';
    xNu-xNl, eye(n)] >= 0;
    for i = 1:n
        Diu(:, :, i) >= 0;
        Dil(:, :, i) >= 0;
        [Diu(:, :, i), -0.5*Delta_x'*e(:, i)-0.5*Diu(:, :, i)'*wd-0.5*Diu(:, :, i)*wu+0.5*Delta_y'*muu(:, i);
            -0.5*e(:, i)'*Delta_x-0.5*wd'*Diu(:, :, i)-0.5*wu'*Diu(:, :, i)'+0.5*muu(:, i)'*Delta_y, -e(:, i)'*Gamma_x*UU(1:N*nu, 1)+e(:, i)'*xNu+wd'*Diu(:, :, i)*wu-muu(:, i)'*(y(ny+1:(N+1)*ny, 1)-Gamma_y*UU(1:N*nu, 1))] >= 0;
        [Dil(:, :, i), 0.5*Delta_x'*e(:, i)-0.5*Dil(:, :, i)'*wd-0.5*Dil(:, :, i)*wu-0.5*Delta_y'*muu(:, i);
            0.5*e(:, i)'*Delta_x-0.5*wd'*Dil(:, :, i)-0.5*wu'*Dil(:, :, i)'-0.5*muu(:, i)'*Delta_y, e(:, i)'*Gamma_x*UU(1:N*nu, 1)-e(:, i)'*xNl+wd'*Dil(:, :, i)*wu+muu(:, i)'*(y(ny+1:(N+1)*ny, 1)-Gamma_y*UU(1:N*nu, 1))] >= 0;
    end
    cvx_end
    XXNu(:, N+1) = xNu;
    XXnl(:, N+1) = xNl;
    X_hat(:, N+1) = (xNu+xNl)/2;
    e = eye(Np*n);
    wu_2 = [xNu; kron(ones(Np, 1), du*ones(nd, 1))];
    wd_2 = [xNl; kron(ones(Np, 1), dl*ones(nd, 1))];
    [Phi, Gamma, Lambda]=prediction_matrices(A, B_d, B_u, Np);
    Delta_x=[Phi, Lambda];
    Gamma_x=Gamma;
    schur_term = inv(Gamma_x'*Q*Gamma_x+R);
    cvx_begin sdp quiet
    cvx_precision high
    variable Diu(Np*nd+n, Np*nd+n, N*n) diagonal;
    variable Dil(Np*nd+n, Np*nd+n, N*n) diagonal;
    variable Dc(Np*nd+n, Np*nd+n) diagonal
    variable g nonnegative;
    variable u(Np*nu, 1);
    minimize(g)
    subject to
    Dc >= 0;
    [Dc-Delta_x'*Q*Delta_x, -0.5*Dc*wu_2-0.5*Dc'*wd_2-Delta_x'*Q*Gamma_x*u, zeros(Np*nd+n, Np*nu);
        (-0.5*Dc*wu_2-0.5*Dc'*wd_2-Delta_x'*Q*Gamma_x*u)', wd_2'*Dc*wu_2+g, u';
        zeros(Np*nu, Np*nd+n), u, schur_term];
    for i = 1:N*n
        Diu(:, :, i) >= 0;
        Dil(:, :, i) >= 0;
        [Diu(:, :, i), -0.5*Delta_x'*e(:, i)-0.5*Diu(:, :, i)'*wd_2-0.5*Diu(:, :, i)*wu_2;
            -0.5*e(:, i)'*Delta_x-0.5*wd_2'*Diu(:, :, i)-0.5*wu_2'*Diu(:, :, i)', -e(:, i)'*Gamma_x*u+e(:, i)'*xh_bar+wd_2'*Diu(:, :, i)*wu_2] >= 0;
        [Dil(:, :, i), 0.5*Delta_x'*e(:, i)-0.5*Dil(:, :, i)'*wd_2-0.5*Dil(:, :, i)*wu_2;
            0.5*e(:, i)'*Delta_x-0.5*wd_2'*Dil(:, :, i)-0.5*wu_2'*Dil(:, :, i)', e(:, i)'*Gamma_x*u-e(:, i)'*xl_bar+wd_2'*Dil(:, :, i)*wu_2] >= 0;
    end
    cvx_end
    UU(N*nu+1: (N+1)*nu, 1) = u(1:nu, 1);
    X1(:, N+2) = A*X1(:, N+1) + B_d*d(:, N+1)+B_u*u(1:nu, 1);
    y((N+1)*ny+1:(N+2)*ny, 1) = C*X1(:, N+2)+D_d*d(:, N+1);
    cost(N) = X1(:, N+1)'*X1(:, N+1) + u(1:nu, 1)'*u(1:nu, 1);
end

for N = (Ne+1):20
    [Phi, Gamma, Lambda]=prediction_matrices(A, B_d, B_u, Ne);
    I = [zeros(1, (Ne-1)), 1];
    Lambda_x = kron(I, eye(n))*Lambda;
    Gamma_x = kron(I, eye(n))*Gamma;
    Delta_x = [kron(I, eye(n))*Phi, Lambda_x];
    Lambda_y = kron(eye(Ne), C)*Lambda + kron(eye(Ne), D_d);
    Gamma_y = kron(eye(Ne), C)*Gamma;
    Delta_y = [kron(eye(Ne), C)*Phi, Lambda_y];
    e = eye(n);
    cvx_begin sdp quiet
    cvx_precision high
    variable Diu(Ne*nd+n, Ne*nd+n, n) diagonal;
    variable Dil(Ne*nd+n, Ne*nd+n, n) diagonal;
    variable g nonnegative;
    variable xNu(n, 1);
    variable xNl(n, 1);
    variable muu(Ne*ny, n);
    minimize(g)
    subject to
    [g, (xNu-xNl)';xNu-xNl, eye(n)] >= 0;
    for i = 1:n
        Diu(:, :, i) >= 0;
        Dil(:, :, i) >= 0;
        [Diu(:, :, i), -0.5*Delta_x'*e(:, i)-0.5*Diu(:, :, i)'*wd-0.5*Diu(:, :, i)*wu+0.5*Delta_y'*muu(:, i);
            -0.5*e(:, i)'*Delta_x-0.5*wd'*Diu(:, :, i)-0.5*wu'*Diu(:, :, i)'+0.5*muu(:, i)'*Delta_y, -e(:, i)'*Gamma_x*UU((N-Ne)*nu+1:N*nu, 1)+e(:, i)'*xNu+wd'*Diu(:, :, i)*wu-muu(:, i)'*(y((N-(Ne-1))*ny+1:(N+1)*ny, 1)-Gamma_y*UU((N-Ne)*nu+1:N*nu, 1))] >= 0;
        [Dil(:, :, i), 0.5*Delta_x'*e(:, i)-0.5*Dil(:, :, i)'*wd-0.5*Dil(:, :, i)*wu-0.5*Delta_y'*muu(:, i);
            0.5*e(:, i)'*Delta_x-0.5*wd'*Dil(:, :, i)-0.5*wu'*Dil(:, :, i)'-0.5*muu(:, i)'*Delta_y, e(:, i)'*Gamma_x*UU((N-Ne)*nu+1:N*nu, 1)-e(:, i)'*xNl+wd'*Dil(:, :, i)*wu+muu(:, i)'*(y((N-(Ne-1))*ny+1:(N+1)*ny, 1)-Gamma_y*UU((N-Ne)*nu+1:N*nu, 1))] >= 0;
    end
    cvx_end
    XXNu(:, N+1) = xNu;
    XXnl(:, N+1) = xNl;
    X_hat(:, N+1) = (xNu+xNl)/2;
    e = eye(Np*n);
    wu_2 = [xNu; kron(ones(Np, 1), du*ones(nd, 1))];
    wd_2 = [xNl; kron(ones(Np, 1), dl*ones(nd, 1))];
    [Phi, Gamma, Lambda] = prediction_matrices(A, B_d, B_u, Np);
    Delta_x = [Phi, Lambda];
    Gamma_x = Gamma;
    cvx_begin sdp quiet
    cvx_precision high
    variable Diu(Np*nd+n, Np*nd+n, N*n) diagonal;
    variable Dil(Np*nd+n, Np*nd+n, N*n) diagonal;
    variable g nonnegative;
    variable u(Np*nu, 1);
    minimize(g)
    subject to
    Dc >= 0;
    [Dc-Delta_x'*Q*Delta_x, -0.5*Dc*wu_2-0.5*Dc'*wd_2-Delta_x'*Q*Gamma_x*u, zeros(Np*nd+n, Np*nu);
        (-0.5*Dc*wu_2-0.5*Dc'*wd_2-Delta_x'*Q*Gamma_x*u)', wd_2'*Dc*wu_2+g, u';
        zeros(Np*nu, Np*nd+n), u, schur_term];
    for i = 1:Np*n
        Diu(:, :, i) >= 0;
        Dil(:, :, i) >= 0;
        [Diu(:, :, i), -0.5*Delta_x'*e(:, i)-0.5*Diu(:, :, i)'*wd_2-0.5*Diu(:, :, i)*wu_2;
            -0.5*e(:, i)'*Delta_x-0.5*wd_2'*Diu(:, :, i)-0.5*wu_2'*Diu(:, :, i)', -e(:, i)'*Gamma_x*u+e(:, i)'*xh_bar+wd_2'*Diu(:, :, i)*wu_2] >= 0;
        [Dil(:, :, i), 0.5*Delta_x'*e(:, i)-0.5*Dil(:, :, i)'*wd_2-0.5*Dil(:, :, i)*wu_2;
            0.5*e(:, i)'*Delta_x-0.5*wd_2'*Dil(:, :, i)-0.5*wu_2'*Dil(:, :, i)', e(:, i)'*Gamma_x*u-e(:, i)'*xl_bar+wd_2'*Dil(:, :, i)*wu_2] >= 0;
    end
    cvx_end
    UU(N*nu+1: (N+1)*nu, 1) = u(1:nu, 1);
    X1(:, N+2) = A*X1(:, N+1)+B_d*d(:, N+1)+B_u*u(1:nu, 1);
    y((N+1)*ny+1:(N+2)*ny, 1) = C*X1(:, N+2)+D_d*d(:, N+1);
    cost(N) = X1(:, N+1)'*X1(:, N+1) + u(1:nu, 1)'*u(1:nu, 1);
end
t1 = toc;
%% Plotting the results
figure(1);
t = 1:21;
for i = 1:n
    subplot(1, n, i);
    plot(t, XXNu(i, :), 'Color','red','LineStyle', '-.');hold on
    plot(t, X1(i, 1:21), 'Color','blue','Marker', '*');
    plot(t, XXnl(i, :), 'Color','black','LineStyle', '-.');
    legend('Upper Bound','Actual State', 'Lower Bound');
    str = sprintf('x_%d',i);
    title(str);
    xlim([1 21]);
    ylim([-10 10]);
    xlabel('Iterations');
    ylabel('State');
    grid on
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For method in chapter 4.2 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% System Rename
B = B_u;
Bd = B_d;
Cy = C;
Dy = 0;
Dyd = D_d;
d_lower = dl;
d_upper = du;

%% System Dimension
nx = 2;
nu = 1;
nd = 1;
ny = 1;

%% Horizon
Nc = 5;
Ne = 5;
N_seq = 21;

%% Cost Weight
Q = eye(nx);
R = eye(nu);
PN = eye(nx);
e = eye(nx);
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

xh = 100*ones(nx, 1);xl = -xh;
uh = 100*ones(nu, 1);ul = -uh;
xh_N = xh;xl_N = -xh_N;
fk = [xh;uh;-xl;-ul];
f_bar = [kron(ones(Nc, 1), fk);xh_N;-xl_N];

%% Stacked System Implementation
[A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Nc);
[Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
[Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
z_bold_bar = zeros(size(Cz_bold, 1), 1);

%% Iterations
x = zeros(nx, N_seq);
x(:,1) = x0;
% Output sequence
y = zeros(ny, N_seq);
% Input sequence
u_out = zeros(nu, N_seq);
cost2 = zeros(1, N_seq);
%% Iterations
x_lower = -5 *ones(nx, 1);
x_upper = 5* ones(nx, 1);
tic;
for k = 1:N_seq
    dk = d(:,k);
    y(:, k) = Cy * x(:,k) + Dyd * dk;
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
        variable g(1) nonnegative;
        variable m(1);
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
        variable g(1) nonnegative;
        variable m(1);
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
    z = Cz * x(:,k) + Dz * u0 + Dzd * dk;
    cost2(1, k) = z'*z;
    x(:, k+1) = A*x(:,k) + B*u0 + Bd*dk;
    u_out(:,k) = u0;
end
t2 = toc;
%% Graphs
% Cost
figure(2);
plot(1:N_seq, cost2, 'Color','blue', 'Marker','*');hold on
plot(t, cost, 'Color','red', 'Marker','*');
xlabel('Iterations');legend('our method', 'Tube-based');
title('Costs');xlim([1 N_seq]);grid on
% Bounds
figure(3);
subplot(1,2,1); 
plot(1:N_seq, x(1, 1:N_seq), 'Color','blue', 'Marker','*');hold on
plot(1:N_seq, X1(1,1:N_seq), 'Color','red', 'Marker','*');
legend('Method in section 4.2', 'Tube-based  MPC');
xlim([1 N_seq]);xlabel('Iterations');ylim([-10 10]);title('x_1');grid on
subplot(1,2,2); 
plot(1:N_seq, x(2, 1:N_seq), 'Color','blue', 'Marker','*');hold on
plot(1:N_seq, X1(2,1:N_seq), 'Color','red', 'Marker','*');
legend('Method in section 4.2', 'Tube-based  MPC');
xlim([1 N_seq]);xlabel('Iterations');ylim([-10 10]);title('x_2');grid on
figure(4);
t_compare = [t1;t2];
b = bar(t_compare);title('Times');xticklabels({'Tube-based  MPC', 'Method in section 4.2'});grid on
xtips1 = b.XEndPoints;ytips1 = b.YEndPoints;labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom');
figure(5);
c = [sum(cost);sum(cost2)];
b = bar(c);title('Costs');xticklabels({'Tube-based  MPC', 'Method in section 4.2'});
xtips1 = b.XEndPoints;ytips1 = b.YEndPoints;labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom');grid on