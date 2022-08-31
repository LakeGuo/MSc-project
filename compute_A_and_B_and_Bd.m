function [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N)
    %% A_bold
    A_bold = A;
    for i = 2:N
        A_bold = [A_bold; A^i];
    end
    %% B_bold
    B_bold = kron(eye(N), B);
    for i = 1:N-1
        B_bold = B_bold + kron(diag(ones(N-i, 1), -i), (A^i)*B);
    end
    %% Bd_bold
    Bd_bold = kron(eye(N), Bd);
    for i = 1:N-1
        Bd_bold = Bd_bold + kron(diag(ones(N-i, 1), -i), (A^i)*Bd);
    end
    Bd_bold = [Bd_bold zeros(N*size(Bd, 1), size(Bd, 2))];
end