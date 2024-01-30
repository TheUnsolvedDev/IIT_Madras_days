// arg[0] filename
// arg[1] port_no > 2048

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>  // system call reasons
#include <sys/socket.h> // sockadress structures
#include <netinet/in.h> //structures for internet domanin addresses

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        error("Port No. not provided!");
    }
    int sock_fd, new_sock_fd, port_no, n;
    char buffer[256];
    struct sockaddr_in server_address, client_address;
    socklen_t client_length;

    // Socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0)
        error("Error Opening the socket\n");

    bzero((char *)&server_address, sizeof(server_address)); // clears the value of the server address
    port_no = atoi(argv[1]);

    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY; // accept any incoming message
    server_address.sin_port = htons(port_no);    // host to network short

    // Binding
    if (bind(sock_fd, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
        error("Binding failed!\n");

    // Listen
    listen(sock_fd, 5); // maximum no of client
    client_length = sizeof(client_address);

    new_sock_fd = accept(sock_fd, (struct sockaddr *)&client_address, &client_length);
    if (new_sock_fd < 0)
        error("Error in accepting!\n");

    for (;;)
    {
        bzero(buffer, 255);
        n = read(new_sock_fd, buffer, 256);
        if (n < 0)
            error("Error on reading");
        printf("Client: %s\n", buffer);
        bzero(buffer, 255);
        fgets(buffer, 255, stdin);

        n = write(new_sock_fd, buffer, strlen(buffer));
        if (n < 0)
            error("Error in writing\n");
        int exit_strat = strncmp("exit", buffer, 4);
        if (exit_strat == 0)
            break;
    }

    close(sock_fd);
    close(new_sock_fd);
    return 0;
}