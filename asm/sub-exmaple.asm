    segment .data
a    dq 100
b    dq 200
diff dq 0

    segment .text
    global main

main:
    push rbp
    mov  rbp, rsp
    sub  rsp, 16

    mov  rax, 10
    sub  [a], rax
    sub  [b], rax
    mov  rax, [b]
    sub  rax, [a]
    mov  [diff], rax
    mov  rax, 0

    leave
    ret
