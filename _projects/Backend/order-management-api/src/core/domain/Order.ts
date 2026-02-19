/**
 * 1. ENUMERATED TYPE (Domain Constants)
 * We define a 'Union Type' to strictly limit what values an order status can have.
 * This prevents typos like 'PANDING' or 'shipped' (lowercase) from entering the system.
 */
export type OrderStatus = 'PENDING' | 'PAID' | 'SHIPPED' | 'CANCELLED';

/**
 * 2. DATA SCHEMA (The Contract)
 * This interface defines the raw data structure of an Order. 
 * We separate 'data' (Props) from 'logic' (Class) to make the code easier to test 
 * and more compatible with database mappers.
 */
export interface OrderProps {
  id: string;               // Unique identifier (usually a UUID)
  customerName: string;     // Name of the person placing the order
  items: string[];          // List of product IDs or names
  totalAmount: number;      // Total price of the order
  status: OrderStatus;      // Must be one of the four types defined above
  createdAt: Date;          // Timestamp of when the order was created
}

/**
 * 3. THE DOMAIN ENTITY (The Logic)
 * The class acts as a "Guard." Its job is to protect the data and ensure 
 * that every Order object in your memory is valid and follows business rules.
 */
export class Order {
  
  /**
   * CONSTRUCTOR
   * 'private props' is a TypeScript shortcut. It creates a private property 
   * called 'props' and assigns the incoming OrderProps to it automatically.
   */
  constructor(private props: OrderProps) {
    // We validate the data immediately. If it's invalid, the object is never created.
    this.validate();
  }

  /**
   * BUSINESS VALIDATION (The "Invariants")
   * This is private because external code shouldn't "trigger" validation; 
   * the class handles its own integrity internally.
   */
  private validate() {
    // Rule 1: A business cannot exist if it loses money on a total.
    if (this.props.totalAmount < 0) {
      throw new Error("The total amount should not be negative/El monto total no puede ser negativo");
    }
    
    // Rule 2: An order without products is technically an empty cart, not an order.
    if (this.props.items.length === 0) {
      throw new Error("The order should have at least one product/La orden debe tener al menos un producto");
    }
  }

  /**
   * SERIALIZATION
   * When sending data to the frontend or a database, we use this method.
   * '{ ...this.props }' creates a shallow copy so the original data stays protected.
   */
  public toJSON() {
    return { ...this.props };
  }

  /**
   * READ-ONLY ACCESSORS (Getters)
   * These allow other parts of the app to see the ID and Status.
   * Since there are no 'setters' (set id() { ... }), these values cannot be 
   * changed from outside the class, ensuring 'Immutability'.
   */
  get id() { 
    return this.props.id; 
  }

  get status() { 
    return this.props.status; 
  }
}
