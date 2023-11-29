#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int value;
    struct node *next;
} node;

void print_list(node *head)
{
    if (head == NULL)
    {
        printf("Empty List\n");
        return;
    }

    node *temp = head;
    do
    {
        printf("%d --> ", temp->value);
        temp = temp->next;
    } while (temp != head);
    printf("(head)\n");
}

void insert_at_beginning(node **head, int value)
{
    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;

    if (*head == NULL)
    {
        new_node->next = new_node;
        *head = new_node;
    }
    else
    {
        new_node->next = (*head)->next;
        (*head)->next = new_node;
    }
}

void insert_at_end(node **head, int value)
{
    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;

    if (*head == NULL)
    {
        new_node->next = new_node;
        *head = new_node;
    }
    else
    {
        new_node->next = (*head)->next;
        (*head)->next = new_node;
        *head = new_node;
    }
}

void insert_at_index(node **head, int index, int value)
{
    if (index <= 0)
    {
        insert_at_beginning(head, value);
        return;
    }

    node *new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL)
    {
        return;
    }
    new_node->value = value;

    if (*head == NULL)
    {
        new_node->next = new_node;
        *head = new_node;
    }
    else
    {
        node *temp = *head;
        for (int i = 0; i < index - 1; ++i)
        {
            temp = temp->next;
        }
        new_node->next = temp->next;
        temp->next = new_node;
        if (index == 1)
        {
            *head = new_node;
        }
    }
}

void delete_at_beginning(node **head)
{
    if (*head != NULL)
    {
        node *temp = (*head)->next;
        if (temp == *head)
        {
            free(*head);
            *head = NULL;
        }
        else
        {
            (*head)->next = temp->next;
            free(temp);
        }
    }
}

void delete_at_end(node **head)
{
    if (*head != NULL)
    {
        node *temp = *head;
        if (temp->next == *head)
        {
            free(temp);
            *head = NULL;
        }
        else
        {
            while (temp->next->next != *head)
            {
                temp = temp->next;
            }
            free(temp->next);
            temp->next = *head;
        }
    }
}

void delete_at_index(node **head, int index)
{
    if (*head != NULL && index >= 0)
    {
        if (index == 0)
        {
            delete_at_beginning(head);
        }
        else
        {
            node *temp = *head;
            for (int i = 0; i < index - 1; ++i)
            {
                temp = temp->next;
            }
            if (temp->next != *head)
            {
                node *to_delete = temp->next;
                temp->next = temp->next->next;
                free(to_delete);
            }
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
