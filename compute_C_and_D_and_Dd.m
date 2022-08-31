function [C_bold, D_bold, Dd_bold] = compute_C_and_D_and_Dd(A, B, Bd, C, C_hat, D, Dd, Dd_hat, N)
    %% C_bold
    C_bold = C;
    for i = 1:N-1
        C_bold = [C_bold; C*A^i];
    end
    C_bold = [C_bold;C_hat*A^N];

    %%D_bold
    D_bold = kron(eye(N), D);
    CHAT = C_hat*B;
    for i = 1:N-1
        D_bold = D_bold + kron(diag(ones(N-i, 1), -i), C*A^(i-1)*B);
        CHAT = [C_hat*(A^i)*B CHAT];
    end
    D_bold = [D_bold; CHAT];

    %% Dd_bold
    Dd_bold = kron(eye(N), Dd);
    CHAT = C_hat*Bd;
    for i = 1:N-1
        Dd_bold = Dd_bold + kron(diag(ones(N-i, 1), -i), C*A^(i-1)*Bd);
        CHAT = [C_hat*(A^i)*Bd CHAT];
    end
    Dd_bold = [Dd_bold; CHAT];
    Dd_bold = [Dd_bold [zeros(N*size(Dd, 1), size(Dd_hat, 2));Dd_hat]];
end
