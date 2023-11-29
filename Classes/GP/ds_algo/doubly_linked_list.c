#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int value;
    struct node *next;
    struct node *prev;
} node;

void print_list(node *head)
{
    node *temp = head;
    while (temp != NULL)
    {
        printf("%d --> ", temp->value);
        temp = temp->next;
    }
    printf("NULL\n");
}

void insert_at_beginning(node **head, int value)
{
    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;
    new_node->prev = NULL;
    new_node->next = *head;

    if (*head != NULL)
    {
        (*head)->prev = new_node;
    }

    *head = new_node;
}

void insert_at_end(node **head, int value)
{
    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;
    new_node->next = NULL;

    if (*head == NULL)
    {
        new_node->prev = NULL;
        *head = new_node;
    }
    else
    {
        node *temp = *head;
        while (temp->next != NULL)
        {
            temp = temp->next;
        }
        temp->next = new_node;
        new_node->prev = temp;
    }
}

void insert_at_index(node **head, int index, int value)
{
    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;

    if (index == 0)
    {
        new_node->prev = NULL;
        new_node->next = *head;

        if (*head != NULL)
        {
            (*head)->prev = new_node;
        }

        *head = new_node;
    }
    else
    {
        node *temp = *head;
        for (int i = 0; i < index - 1 && temp != NULL; ++i)
        {
            temp = temp->next;
        }
        if (temp == NULL)
        {
            free(new_node);
            return;
        }
        new_node->next = temp->next;
        new_node->prev = temp;
        if (temp->next != NULL)
        {
            temp->next->prev = new_node;
        }
        temp->next = new_node;
    }
}

void delete_at_beginning(node **head)
{
    if (*head != NULL)
    {
        node *temp = *head;
        *head = (*head)->next;

        if (*head != NULL)
        {
            (*head)->prev = NULL;
        }

        free(temp);
    }
}

void delete_at_end(node **head)
{
    if (*head != NULL)
    {
        if ((*head)->next == NULL)
        {
            free(*head);
            *head = NULL;
        }
        else
        {
            node *temp = *head;
            while (temp->next->next != NULL)
            {
                temp = temp->next;
            }
            free(temp->next);
            temp->next = NULL;
        }
    }
}

void delete_at_index(node **head, int index)
{
    if (*head != NULL && index >= 0)
    {
        if (index == 0)
        {
            node *temp = *head;
            *head = (*head)->next;

            if (*head != NULL)
            {
                (*head)->prev = NULL;
            }

            free(temp);
        }
        else
        {
            node *temp = *head;
            for (int i = 0; i < index - 1 && temp != NULL; ++i)
            {
                temp = temp->next;
            }
            if (temp == NULL || temp->next == NULL)
            {
                return;
            }
            node *to_delete = temp->next;
            temp->next = temp->next->next;

            if (temp->next != NULL)
            {
                temp->next->prev = temp;
            }

            free(to_delete);
        }
    }
}

int main()
{
    node *head = NULL;

    for (int i = 0; i < 10; i++)
        insert_at_beginning(&head, i);

    for (int i = 0; i < 10; i++)
        insert_at_end(&head, i);

    for (int i = 0; i < 10; i++)
        insert_at_index(&head, i, 100);

    for (int i = 0; i < 10; i++)
        delete_at_beginning(&head);

    print_list(head);
    return 0;
}
