// arg[0] filename
// arg[1] server_ipaddress
// arg[2] port_no

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>  // system call reasons
#include <sys/socket.h> // sockadress structures
#include <netinet/in.h> //structures for internet domanin addresses
#include <netdb.h>

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    int sock_fd, port_no, n;
    struct sockaddr_in server_address;
    struct hostent *server;
    char buffer[255];

    if (argc < 3)
    {
        fprintf(stderr, "usage: %s hostname port\n", argv[0]);
        exit(1);
    }
    port_no = atoi(argv[2]);
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0)
        error("Error opening the socket\n");

    server = gethostbyname(argv[1]);
    if (server == NULL)
        error("No such host\n");

    bzero((char *)&server_address, sizeof(server_address));
    server_address.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&server_address.sin_addr.s_addr, server->h_length);
    server_address.sin_port = htons(port_no);

    if (connect(sock_fd, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
        error("Connection Failed\n");

    for (;;)
    {
        bzero(buffer, 255);
        printf(":>");
        fgets(buffer, 255, stdin);
        n = write(sock_fd, buffer, strlen(buffer));
        if (n < 0)
            error("Error in writing!\n");

        bzero(buffer, 255);
        n = read(sock_fd, buffer, 255);
        if (n < 0)
            error("Error on reading!\n");
        printf("Server: %s", buffer);

        int exit_strat = strncmp("exit", buffer, 4);
        if (exit_strat == 0)
            break;
    }
    close(sock_fd);
    return 0;
}